# vendor/ironrdp-egfx — divergence log

Local fork of ironrdp-egfx 0.3.0, **vendored 2026-10-22 from upstream
Devolutions/IronRDP@a5d1c682** (the current pin). Upstream files are copied
verbatim; the only changes are pure additions in `src/server.rs`. Has a standalone
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

(2) `GraphicsPipelineServer::send_bitmap_tiles_frame` (NOT upstreamed): one
    StartFrame / WireToSurface1... / EndFrame with a caller-chosen codec and
    destination per tile. Upstream's `send_planar_frame` and
    `send_uncompressed_frame` always draw one tile at the surface origin, and
    `MixedTilePayload` has no Planar or Uncompressed variant. macrdp's EGFX
    fast path (`src/egfx_fast_path.rs`) sends small changed rects this way
    ahead of the H.264 frame. Upstream candidate: a destination on the
    planar/uncompressed sends, or the two variants on `MixedTilePayload`.
(3) `GraphicsPipelineServer::send_avc444_frame_to_rect` (NOT upstreamed): an
    AVC444 v1 send that takes the LC value (`Encoding`) and the destination.
    The vendored `send_avc444_frame` can only send LC 0 and 1 and derives the
    destination the same inclusive way as (1). macrdp's opt-in lazy chroma
    (`MACRDP_AVC444=lazy`, `src/lazy_chroma.rs`) sends motion as LC 1 and the
    auxiliary view alone as LC 2 once the screen is idle; LC 2 carries a zero
    `cbAvc420EncodedBitstream1` as the spec requires (upstream's encoder does
    the same since `da4d45a1`, which has LC 2 only for v2).

Retire this vendor dir once upstream offers an equivalent and the pin moves past it.
