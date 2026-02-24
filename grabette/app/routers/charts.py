"""Live sensor chart endpoints — uPlot charts served as HTML for iframe embedding."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["charts"])

_UPLOT_HEAD = """\
<meta charset="utf-8">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/uplot@1.6.32/dist/uPlot.min.css">
<script src="https://cdn.jsdelivr.net/npm/uplot@1.6.32/dist/uPlot.iife.min.js"></script>
<style>
body{margin:0;background:transparent;overflow:hidden}
.u-title{color:#ccc;font-size:11px}
.u-legend .u-label{color:#aaa}
.u-legend .u-value{color:#ccc}
.u-legend{font-size:10px}
</style>"""

IMU_CHART_HTML = f"""\
<!DOCTYPE html>
<html><head>
{_UPLOT_HEAD}
</head><body>
<div id="accel"></div>
<div id="gyro" style="margin-top:2px"></div>
<script>
var MAXLEN=30;
function roll(a,v){{a.push(v);if(a.length>MAXLEN)a.shift();}}
var iT=[],ax=[],ay=[],az=[],gx=[],gy=[],gz=[],t0=null;

(function init(){{
  var w=document.body.clientWidth||400;
  var h=Math.floor((window.innerHeight-6)/2);
  function opts(title,yLbl,ser){{
    return{{width:w,height:h,title:title,
      cursor:{{show:false}},legend:{{show:true,live:false}},
      scales:{{x:{{time:false}}}},series:[{{}}].concat(ser),
      axes:[{{stroke:'#888',grid:{{stroke:'#333'}},size:30}},
            {{stroke:'#888',grid:{{stroke:'#333'}},label:yLbl,size:50}}]}};
  }}
  var aC=new uPlot(opts('Accelerometer','m/s\\u00b2',[
    {{label:'X',stroke:'#e55',width:1}},
    {{label:'Y',stroke:'#5b5',width:1}},
    {{label:'Z',stroke:'#55e',width:1}}]),
    [[],[],[],[]],document.getElementById('accel'));
  var gC=new uPlot(opts('Gyroscope','rad/s',[
    {{label:'X',stroke:'#e55',width:1}},
    {{label:'Y',stroke:'#5b5',width:1}},
    {{label:'Z',stroke:'#55e',width:1}}]),
    [[],[],[],[]],document.getElementById('gyro'));

  new ResizeObserver(function(){{
    var nw=document.body.clientWidth;
    var nh=Math.floor((window.innerHeight-6)/2);
    aC.setSize({{width:nw,height:nh}});
    gC.setSize({{width:nw,height:nh}});
  }}).observe(document.body);

  setInterval(function(){{
    fetch('/api/state').then(function(r){{return r.ok?r.json():null;}})
    .then(function(s){{
      if(!s||!s.imu)return;
      var now=performance.now()/1000;
      if(t0===null)t0=now;var t=now-t0;
      var a=s.imu.accel,g=s.imu.gyro;
      roll(iT,t);roll(ax,a[0]);roll(ay,a[1]);roll(az,a[2]);
      roll(gx,g[0]);roll(gy,g[1]);roll(gz,g[2]);
      aC.setData([iT.slice(),ax.slice(),ay.slice(),az.slice()]);
      gC.setData([iT.slice(),gx.slice(),gy.slice(),gz.slice()]);
    }}).catch(function(){{}});
  }},500);
}})();
</script>
</body></html>"""

ANGLE_CHART_HTML = f"""\
<!DOCTYPE html>
<html><head>
{_UPLOT_HEAD}
</head><body>
<div id="angle"></div>
<script>
var MAXLEN=30;
function roll(a,v){{a.push(v);if(a.length>MAXLEN)a.shift();}}
var aT=[],pr=[],di=[],t0=null;

(function init(){{
  var w=document.body.clientWidth||400;
  var h=window.innerHeight||120;
  var nC=new uPlot({{
    width:w,height:h,title:'Angle Sensors',
    cursor:{{show:false}},legend:{{show:true,live:false}},
    scales:{{x:{{time:false}}}},
    series:[{{}},
      {{label:'Proximal',stroke:'#4488cc',width:1.5}},
      {{label:'Distal',stroke:'#cc8844',width:1.5}}],
    axes:[{{stroke:'#888',grid:{{stroke:'#333'}},size:30}},
          {{stroke:'#888',grid:{{stroke:'#333'}},label:'Degrees',size:50}}]
  }},[[],[],[]],document.getElementById('angle'));

  new ResizeObserver(function(){{
    nC.setSize({{width:document.body.clientWidth,height:window.innerHeight}});
  }}).observe(document.body);

  setInterval(function(){{
    fetch('/api/state').then(function(r){{return r.ok?r.json():null;}})
    .then(function(s){{
      if(!s||!s.angle)return;
      var now=performance.now()/1000;
      if(t0===null)t0=now;var t=now-t0;
      var p=s.angle.proximal*180/Math.PI;
      var d=s.angle.distal*180/Math.PI;
      roll(aT,t);roll(pr,p);roll(di,d);
      nC.setData([aT.slice(),pr.slice(),di.slice()]);
    }}).catch(function(){{}});
  }},500);
}})();
</script>
</body></html>"""


@router.get("/charts/imu")
async def imu_chart():
    """Serve the IMU chart page (accelerometer + gyroscope)."""
    return HTMLResponse(content=IMU_CHART_HTML)


@router.get("/charts/angle")
async def angle_chart():
    """Serve the angle sensor chart page."""
    return HTMLResponse(content=ANGLE_CHART_HTML)
