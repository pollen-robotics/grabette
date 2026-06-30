/**
 * Offline RTABMapVIO — processes a session recorded with record_vslam.py.
 *
 * No depthai dependency: reads PNG + CSV files and drives RTABMap Odometry
 * directly, with the exact same logic as RTABMapVIO::syncCB / initialize().
 *
 * Usage:
 *   ./offline_vslam <recording_dir> [output_poses.csv]
 *
 * Build (standalone, no cmake needed):
 *   See build_offline_vslam.sh
 */

#include <rtabmap/core/IMU.h>
#include <rtabmap/core/Odometry.h>
#include <rtabmap/core/OdometryInfo.h>
#include <rtabmap/core/Parameters.h>
#include <rtabmap/core/SensorData.h>
#include <rtabmap/core/StereoCameraModel.h>
#include <rtabmap/core/Transform.h>
#include <rtabmap/utilite/ULogger.h>

#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>

#include <Eigen/Geometry>

#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static std::vector<std::string> splitCsv(const std::string& line) {
    std::vector<std::string> tok;
    std::stringstream ss(line);
    std::string t;
    while(std::getline(ss, t, ',')) tok.push_back(t);
    return tok;
}

static std::string frameName(int idx) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%06d.png", idx);
    return buf;
}

int main(int argc, char** argv) {
    if(argc < 2) {
        std::cerr << "Usage: offline_vslam <recording_dir> [output_poses.csv]\n";
        return 1;
    }

    const std::string dir        = std::string(argv[1]);
    const std::string posesPath  = (argc > 2) ? argv[2] : dir + "/poses.csv";
    const auto path = [&](const std::string& rel) { return dir + "/" + rel; };

    ULogger::setType(ULogger::kTypeConsole);
    ULogger::setLevel(ULogger::kWarning);

    // -----------------------------------------------------------------------
    // 1. Calibration — same initialisation as RTABMapVIO::initialize()
    // -----------------------------------------------------------------------
    nlohmann::json calib;
    {
        std::ifstream f(path("calib_offline.json"));
        if(!f) { std::cerr << "Cannot open calib_offline.json\n"; return 1; }
        f >> calib;
    }

    const int    width    = calib["width"];
    const int    height   = calib["height"];
    const double fx       = calib["fx"],  fy = calib["fy"];
    const double cx       = calib["cx"],  cy = calib["cy"];
    const double baseline = calib["baseline"];

    rtabmap::Transform opticalTransform(0,0,1,0, -1,0,0,0, 0,-1,0,0);
    rtabmap::Transform localTransform = rtabmap::Transform::getIdentity() * opticalTransform.inverse();

    // Camera-frame fix-up matrix.
    //
    // Triangulated empirically from two observations:
    //  - `pose * opticalTransform` produced an FRD camera frame
    //    (forward=+X, right=+Y, down=+Z).
    //  - `pose * [[0,0,1],[1,0,0],[0,1,0]]` produced a BLD frame
    //    (forward=-X, right=-Y, down=+Z) — i.e. the previous frame
    //    rotated 180° around its Z (down) axis.
    //
    // Neither is OpenCV optical RDF (right, down, forward). To go from BLD →
    // RDF, right-multiply by R_BLD_from_RDF = [[0,0,-1],[-1,0,0],[0,1,0]].
    // Composing that with the BLD-producing matrix yields the matrix below.
    //
    // The composition is no longer a simple "optical/baselink" change of basis
    // because rtabmap's effective output frame doesn't match either of the
    // standard ROS conventions we assumed — this matrix encodes the specific
    // rotation needed to land on OpenCV optical.
    const rtabmap::Transform R_camera_fix(
        0, -1, 0, 0,
        0, 0, -1, 0,
        1, 0, 0, 0);

    rtabmap::StereoCameraModel model(
        "oak", fx, fy, cx, cy, baseline, localTransform, cv::Size(width, height));

    auto& itc = calib["imu_to_cam"];
    // imu_to_cam from depthai: takes IMU body vectors to camera optical vectors.
    rtabmap::Transform imuLocalTransform(
        (double)itc[0][0], (double)itc[0][1], (double)itc[0][2], (double)itc[0][3] * 0.01,
        (double)itc[1][0], (double)itc[1][1], (double)itc[1][2], (double)itc[1][3] * 0.01,
        (double)itc[2][0], (double)itc[2][1], (double)itc[2][2], (double)itc[2][3] * 0.01);
    // RTAB-Map's IMU.localTransform: ROS convention "pose of IMU IN base"
    // (Option A) = T_base←imu. Compose as:
    //   T_base←imu = T_base←optical * T_optical←imu
    //              = opticalTransform * imu_to_cam (loaded)
    // Empirical debug print of the previous Option B form
    // (imu_to_cam.inverse() * localTransform) had roll=π, which would have
    // caused RTAB-Map to see the camera upside-down at t=0 — that matched
    // the observed world-frame tilt.
    imuLocalTransform = opticalTransform * imuLocalTransform;

    // -----------------------------------------------------------------------
    // 2. IMU buffers — same structure as RTABMapVIO::Impl
    // -----------------------------------------------------------------------
    std::map<double, cv::Vec3f> accBuffer, gyroBuffer;
    auto loadImu = [&](const std::string& file, std::map<double, cv::Vec3f>& buf) {
        std::ifstream f(path(file));
        if(!f) { std::cerr << "Cannot open " << file << "\n"; return; }
        std::string line;
        std::getline(f, line); // header
        while(std::getline(f, line)) {
            auto t = splitCsv(line);
            if(t.size() < 4) continue;
            double stamp = std::stod(t[0]) * 1e-9;
            buf[stamp] = cv::Vec3f(std::stof(t[1]), std::stof(t[2]), std::stof(t[3]));
        }
    };
    loadImu("imu_acc.csv",  accBuffer);
    loadImu("imu_gyro.csv", gyroBuffer);

    // Rotation buffer (qx, qy, qz, qw) from BNO086 rotation_vector (Eigen for slerp).
    std::map<double, Eigen::Quaternionf> rotBuffer;
    {
        std::ifstream f(path("imu_rotation.csv"));
        if(f) {
            std::string line;
            std::getline(f, line); // header
            while(std::getline(f, line)) {
                auto t = splitCsv(line);
                if(t.size() < 5) continue;
                double stamp = std::stod(t[0]) * 1e-9;
                // Eigen::Quaternionf(w, x, y, z) — note the W-first constructor!
                rotBuffer.emplace(stamp, Eigen::Quaternionf(
                    std::stof(t[4]), std::stof(t[1]), std::stof(t[2]), std::stof(t[3])));
            }
        }
    }
    std::cout << "IMU: " << accBuffer.size() << " acc, "
              << gyroBuffer.size() << " gyro, "
              << rotBuffer.size() << " rotation samples\n";

    // -----------------------------------------------------------------------
    // 3. Frame timestamps
    // -----------------------------------------------------------------------
    std::vector<std::pair<int, double>> frames; // (idx, stamp_seconds)
    {
        std::ifstream f(path("timestamps.csv"));
        if(!f) { std::cerr << "Cannot open timestamps.csv\n"; return 1; }
        std::string line;
        std::getline(f, line); // header
        while(std::getline(f, line)) {
            auto t = splitCsv(line);
            if(t.size() < 2) continue;
            frames.emplace_back(std::stoi(t[0]), std::stod(t[1]) * 1e-9);
        }
    }
    std::cout << "Frames: " << frames.size() << "\n";

    // -----------------------------------------------------------------------
    // 4. Odometry
    // -----------------------------------------------------------------------
    rtabmap::ParametersMap params;
    params[rtabmap::Parameters::kOdomResetCountdown()] = "30";
    auto odom = rtabmap::Odometry::create(params);

    // -----------------------------------------------------------------------
    // 5. Process — exact same logic as RTABMapVIO::syncCB
    // -----------------------------------------------------------------------
    std::ofstream posesOut(posesPath);
    posesOut << "timestamp_s,dx,dy,dz,dqx,dqy,dqz,dqw,lost,inliers,inliers_ratio,variance,features\n";

    rtabmap::Transform prevPose = rtabmap::Transform::getIdentity();

    for(auto& [idx, stamp] : frames) {
        cv::Mat gray  = cv::imread(path("frames/") + frameName(idx), cv::IMREAD_GRAYSCALE);
        cv::Mat depth = cv::imread(path("depth/")  + frameName(idx), cv::IMREAD_UNCHANGED);

        if(gray.empty() || depth.empty()) {
            std::cerr << "Missing frame " << idx << ", skipping\n";
            continue;
        }

        rtabmap::SensorData data(gray, depth, model.left(), idx, stamp);

        // IMU interpolation — mirrors RTABMapVIO::syncCB exactly
        if(!accBuffer.empty() && !gyroBuffer.empty()
           && accBuffer.rbegin()->first  >= stamp
           && gyroBuffer.rbegin()->first >= stamp) {

            cv::Vec3d acc(0,0,0), gyro(0,0,0);

            auto interp = [&](std::map<double, cv::Vec3f>& buf, cv::Vec3d& out) {
                auto iterB = buf.lower_bound(stamp);
                if(iterB != buf.end()) {
                    auto iterA = iterB;
                    if(iterA != buf.begin()) --iterA;
                    if(iterA == iterB || stamp == iterB->first) {
                        out = iterB->second;
                    } else if(stamp > iterA->first && stamp < iterB->first) {
                        float t = (stamp - iterA->first) / (iterB->first - iterA->first);
                        out = iterA->second + t * (iterB->second - iterA->second);
                    }
                    buf.erase(buf.begin(), iterB);
                }
            };

            interp(accBuffer,  acc);
            interp(gyroBuffer, gyro);

            // Interpolate rotation_vector via SLERP. Pre-clean older entries
            // to keep the buffer bounded (mirrors the acc/gyro interp pattern).
            cv::Vec4d orient(0,0,0,1);
            bool haveOrient = false;
            Eigen::Quaternionf qOut;
            if(!rotBuffer.empty()) {
                auto iterB = rotBuffer.lower_bound(stamp);
                if(iterB == rotBuffer.end()) {
                    --iterB;
                    qOut = iterB->second;
                    haveOrient = true;
                } else if(iterB != rotBuffer.begin()) {
                    auto iterA = std::prev(iterB);
                    float t = float((stamp - iterA->first) / (iterB->first - iterA->first));
                    qOut = iterA->second.slerp(t, iterB->second);
                    haveOrient = true;
                    rotBuffer.erase(rotBuffer.begin(), iterA);
                } else {
                    qOut = iterB->second;
                    haveOrient = true;
                }
            }
            if(haveOrient) {
                // NOTE: rtabmap::IMU expects orientation Vec4d as (w, x, y, z),
                // NOT the ROS-standard (x, y, z, w). Empirically verified by
                // A/B testing both conventions — only wxyz gives an initial
                // pose with roll≈0 (camera right-side-up). This is undocumented
                // in rtabmap and surprising; deviating from this order produces
                // a camera that RTAB-Map thinks is rotated 180° at t=0.
                orient = {qOut.w(), qOut.x(), qOut.y(), qOut.z()};
            }

            if(haveOrient) {
                data.setIMU(rtabmap::IMU(
                    orient, cv::Mat::eye(3,3,CV_64FC1),
                    gyro,   cv::Mat::eye(3,3,CV_64FC1),
                    acc,    cv::Mat::eye(3,3,CV_64FC1),
                    imuLocalTransform));
            } else {
                data.setIMU(rtabmap::IMU(
                    gyro, cv::Mat::eye(3,3,CV_64FC1),
                    acc,  cv::Mat::eye(3,3,CV_64FC1),
                    imuLocalTransform));
            }
        }

        rtabmap::OdometryInfo info;
        auto pose = odom->process(data, &info);
        // Re-express camera's local axes into OpenCV optical RDF (project's
        // standard camera convention). World frame is unchanged (Z-up,
        // gravity-aligned from IMU init). Deltas computed downstream are in
        // the optical local frame.
        pose = pose * R_camera_fix;

        rtabmap::Transform delta = prevPose.inverse() * pose;
        prevPose = pose;

        auto q = delta.getQuaternionf();
        posesOut << stamp
            << "," << delta.x() << "," << delta.y() << "," << delta.z()
            << "," << q.x()     << "," << q.y()     << "," << q.z() << "," << q.w()
            << "," << (info.lost ? 1 : 0)
            << "," << info.reg.inliers
            << "," << info.reg.inliersRatio
            << "," << info.reg.covariance.at<double>(0,0)
            << "," << info.features
            << "\n";

        if(idx % 30 == 0)
            std::cout << "Frame " << idx << "  delta=" << delta.prettyPrint() << "\n";
    }

    std::cout << "Poses written to " << posesPath << "\n";
    return 0;
}
