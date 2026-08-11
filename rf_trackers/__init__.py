"""
Vendored subset of Roboflow `trackers` 2.6.0 (Apache-2.0), renamed.

EyeCU owns a top-level package called `trackers`, so the upstream
package cannot be installed alongside it without the import being
decided by path order. Only the modules CBIoUTracker needs are copied
here, with `trackers.` rewritten to `rf_trackers.` in their imports
and nothing else touched.

Only CBIoUTracker is exported. BoTSORT modules are present because
CBIoUTracker inherits from BoTSORTTracker; they are an implementation
dependency, not a second production tracker.

Provenance, hashes and licence: see VENDOR_PROVENANCE.json and LICENSE.
"""

from rf_trackers.core.cbiou.tracker import CBIoUTracker

__all__ = ['CBIoUTracker']
__vendored_from__ = 'trackers==2.6.0'
