"""Casquette fleet client — the device-side of the cloud fleet/multi-device
system (relay poll loop, T0 capture scheduler, cancel registry, HF auth).

COPY-FIRST PROTOTYPE: these modules are copied, with minimal deviations, from
grabette's fleet client (develop @0cb3453). The boundary is deliberately clean
(the fleet code depends only on injected config/providers + a Backend/record
interface) so this subpackage can later be extracted into a shared package that
both grabette and casquette import. Until then, keep deviations from grabette
small and documented in each module header. See the casquette-sync-port memory.
"""
