# vendor/ironrdp-egfx — divergence log

Local fork of ironrdp-egfx 0.3.0, **vendored 2026-10-22 from upstream
Devolutions/IronRDP@a5d1c682** (the current pin). Upstream files are copied
verbatim; the only change is a pure addition in `src/server.rs`. Has a standalone
`[patch.crates-io]` (core/dvc/graphics/pdu → a5d1c682) for isolated build, ignored
in the macrdp workspace (root `[patch]` wins). Keep that rev in sync when bumping
the IronRDP pin. `rustfmt.toml` mirrors the IronRDP workspace config so a
reformat doesn't churn the crate away from upstream. The lib is `test = false`;
the region math it carries is tested from macrdp (`src/damage.rs`, `src/h264.rs`).

Patch wiring: only `[patch.crates-io]` in the root Cargo.toml. No crate macrdp
pulls from the IronRDP git workspace depends on ironrdp-egfx by path (the vendored
ironrdp-server names it as a crates.io dep), so the git-source patch the vendored
ironrdp-dvc needs isn't required here.

## Divergence

(1) `GraphicsPipelineServer::send_avc420_frame_to_rect` (NOT upstreamed): a copy
    of `send_avc420_frame` that takes the WireToSurface1 `destination` instead of
    deriving it from the regions. Upstream's `compute_dest_rect` treats
    `Avc420Region` bounds as inclusive and adds 1 to the bounding box, while
    the wire `RDPGFX_RECT16` (regionRects and destRect alike) is right/bottom
    exclusive: FreeRDP rejects `left >= right` and converts `[left, right)`.
    So upstream either leaves the last column and row of a whole-surface region
    unpainted on FreeRDP (inclusive `width - 1`) or, with exclusive regions
    reaching the edge, emits a destRect one pixel past the surface, which
    FreeRDP's `is_within_surface` rejects. macrdp sends the frame's changed
    rects (exclusive, in surface coordinates) as regions and the whole surface
    as the destination, as FreeRDP's own shadow server does. Upstream
    candidate: let the caller pass the destination, or make `Avc420Region`
    exclusive.

Retire this vendor dir once upstream offers an equivalent and the pin moves past it.
