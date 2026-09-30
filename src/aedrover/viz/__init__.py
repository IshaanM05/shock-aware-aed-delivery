"""Cinematic rendering of recorded episodes (optional; the simulator never imports this package).

Episodes are recorded once on the CPU (``recording``), then replayed through a dressed copy of the
physics model (``render_model``) by a PBR or classic renderer (``backend``), post-processed
(``post``) and cut into shots (``camera``, ``film``). Install the extras with ``pip install -e ".[viz]"``.
"""
