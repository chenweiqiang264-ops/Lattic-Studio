# TPMS Lattice Design Context

This context covers implicit TPMS lattice generation inside a watertight design domain, including transitions between G and D lattice fields.

## Block-workflow terms

These terms describe the agreed target workflow; their presence does not imply
that the existing workbench has implemented it.

**Design operation block**:
An independently identified modeling operation within one Design document,
with its own parameters, input references, output, and computation status.
Multiple blocks may represent the same operation type.

**Block input reference**:
An explicit reference to an upstream block's output within the same Design
document. Cross-document reuse creates an independent copy.

**Outdated block result**:
A retained result whose generating parameters or upstream dependencies have
changed; it remains inspectable as an old result until explicitly updated.

**Archived operation block**:
A recoverably removed Design operation block whose downstream configurations
are retained with missing input references until restored or reassigned.

**Transition periodic-field input**:
A reference to a lattice block's unclipped periodic field, preserving its
cell parameters, placement, and gradient definition independently of that
lattice block's design-domain clipping.

**Transition design-domain input**:
The explicitly selected geometry that bounds a transition block's blended
result; it may differ from either operand lattice's Design domain.

**Selected operation block**:
The block currently targeted for parameter editing and inspection; selection
does not change the visibility of other blocks' outputs.

**Block output visibility**:
The independent display state of a block's visual output. An explicit
isolate-view command shows only the targeted output without changing geometry
or its dependencies.

## Design-domain terms

**Design domain**:
The independent geometric volume that limits where the generated lattice may
exist. It may be an imported watertight STL or a parameterized analytic
implicit body such as a sphere, cylinder, or box. Its authoritative definition
is the selected evaluator; an STL mesh or preview mesh is only a source or
derived representation.

**Design document**:
An isolated modeling document that owns its geometry, operation blocks,
parameters, results, and display caches. In the agreed block workflow it may
contain multiple Design domains, explicitly referenced by individual lattice
blocks, without sharing mutable state across documents.

**Lattice design-domain input**:
The explicit geometry reference defining one lattice block's filling boundary;
different lattice blocks in the same Design document may reference different
Design domains.

**Design workspace**:
The container for multiple independent Design documents and the active-document
selection. It coordinates document creation, removal, and switching without
making fields, parameters, results, or caches global across documents.

**Active design**:
The single Design document whose domain, field objects, results, and
manipulators are rendered and may receive generation, inspection, or export
commands. Other documents remain retained by the Design workspace but are not
rendered, picked, or mutated.

**Analytic design domain**:
A sphere, cylinder, or box selected as a Design document's Design domain. Its
parameterized signed-distance evaluator is authoritative throughout lattice
generation; a surface mesh is derived only for display or export.
_Avoid_: Converted primitive STL, imported mesh domain

**Dual-role analytic primitive**:
An Analytic design domain that is also referenced as a transition driver field
in the same Design document. It remains one authoritative SDF; transforming it
invalidates both the Design domain dependents and the transition dependents.

**Design creation source**:
The explicit origin of a new Design document: either an imported STL or an
Analytic design domain. A primitive retained only in the Field-object scene is
not a Design creation source.

**Single-source Design domain**:
A Design domain defined by one imported STL or one Analytic design domain.
This describes an individual domain's source, not a limit on the number of
domains owned by a Design document.

**Editable Analytic design domain**:
An Analytic design domain whose position, orientation, and dimensions may be
edited interactively. An edit immediately changes the authoritative SDF and
dependent design guidance, clears its dependent results, and never silently
starts a new lattice generation.

**Archived design**:
A soft-deleted Design document retained by the Design workspace outside its
active design list. It is not rendered, picked, generated, or exported, but
may be restored with its owned state intact or permanently removed by an
explicit later action.

**Empty Design workspace**:
A Design workspace with no active Design documents. It has no Active design
and renders no design scene until a document is created or an Archived design
is restored. A newly launched application starts with an Empty Design workspace;
example geometry is created only through an explicit command.

**Persistent Design workspace**:
A Design workspace whose active and Archived designs, authoritative geometry
definitions, parameters, references, and derived-result provenance survive
application restart. Disposable sampled-field and renderer caches are rebuilt
on demand and are not persistent design state.

**Packaged mesh source**:
An imported design-domain STL or custom unit-cell STL copied into the Design
workspace's managed assets before persistence. Design documents reference the
managed copy so moving or sharing the workspace cannot break their source
geometry.

**Persisted derived mesh**:
A generated or exported STL retained as a managed Design workspace asset with
its generation provenance. It may be shown immediately after reopening, while
the corresponding sampled implicit field and renderer resources are rebuilt on
demand rather than serialized.

**Dirty Design workspace**:
A Persistent Design workspace with unsaved edits. Edits mark it dirty, while
managed assets and metadata are written only through an explicit save command,
close confirmation, or workspace-switch confirmation rather than on every
interactive parameter change.

**Design revision**:
The monotonically changing state version of a Design document. A background
generation task captures its document identifier and Design revision at launch;
its result belongs only to that document and is stale when the revision has
changed before completion.

**TPMS field**:
An implicit scalar field describing one periodic lattice family, normalized to a physical signed-distance-like field in millimetres.

**Implicit volume**:
An implicit solid whose boundary is the zero set of its authoritative implicit evaluator.

**Implicit evaluator**:
The authoritative, resolution-independent definition that returns an implicit value for any requested point within its bounds.
_Avoid_: Voxel array, render cache

**Sampled field cache**:
A bounded-resolution scalar grid derived from an implicit evaluator for display or mesh extraction and safe to discard and rebuild.
_Avoid_: Authoritative implicit body

**Resident implicit render layer**:
A sampled field's reusable VTK image, GPU mapper, and volume actor. Hiding a layer or temporarily viewing a mesh keeps these resources resident within the display-memory budget, while geometry changes create a new layer identity.
_Avoid_: Authoritative implicit body, unbounded GPU cache

**Display voxel size**:
The sampling spacing of an interactive sampled-field cache; it affects preview fidelity and GPU memory but not authoritative geometry or export quality.
_Avoid_: Export voxel size, geometry tolerance

**Display memory budget**:
The GPU-oriented memory allowance used to bound an interactive sampled-field cache and trigger preview-only resolution fallback.
_Avoid_: Generation voxel limit

**Preview implicit shell**:
A display-only implicit shell derived from a non-watertight surface without claiming a reliable inside/outside volume.
_Avoid_: Design domain, boolean-ready implicit volume

**Open implicit section**:
An interactive clipping plane that hides one side of an implicit surface without generating a cap face on the cut boundary.
_Avoid_: Mesh clip, capped section mesh

**Implicit generation result**:
The evaluator-backed TPMS result shown by default after generation; a surface mesh is a later derived artifact for export or mesh-only operations.
_Avoid_: Generated STL, marching-cubes result

**Unclipped lattice feature body**:
The evaluator-backed periodic lattice field before design-domain restriction. An implicit generation result retains this dependency so shell/lattice fusion can evaluate the design-domain SDF once per point batch and reuse it for lattice clipping, shell construction, and the final domain bound.
_Avoid_: Display cache, exported lattice mesh

**Dependency invalidation**:
The rule that changes to a design domain invalidate all dependent implicit results, while changes to one lattice or transition invalidate only that result and its display cache.
_Avoid_: Recompute everything

**Domain overlay**:
The independently rendered design-domain implicit body that provides spatial context around the lattice, currently shown as an opaque layer.
_Avoid_: Boolean-only domain

**Automatic numerical backend**:
The compute interfaces that select CUDA for G, D, transition, design-domain SDF, implicit intersection, and Marching Cubes when available and fall back to CPU adapters after any GPU runtime failure.
_Avoid_: GPU renderer, assumed CUDA for every operation

**Automatic rendering backend**:
The display path that probes the active OpenGL context, uses hardware GPU zero-isosurface ray casting when supported, and reports software OpenGL or unsupported volume mapping as an explicit degradation.
_Avoid_: CUDA numerical backend, assumed GPU rendering

**PBR-style implicit material**:
Metallic and roughness controls mapped to the lit GPU implicit-isosurface model. Polygon meshes use VTK's native PBR interpolation; implicit volumes remain ray-cast fields and therefore use a calibrated material approximation rather than a temporary display mesh.
_Avoid_: Extracted PBR preview mesh, unlit volume

**Studio lighting rig**:
A model-centred key, fill, rim, and low fill arrangement used to reveal surface form without flattening the model with a dominant camera headlight.
_Avoid_: Camera-only lighting, decorative lighting

**Derived surface mesh**:
A triangulated representation produced from an implicit volume only when a mesh operation, inspection, or export requires it.
_Avoid_: Final model, authoritative STL

**Derived-mesh display proxy**:
A cached, quality-bounded VTK surface used only to keep large STL interaction responsive. It never replaces or modifies the derived surface mesh used for inspection and export.
_Avoid_: Simplified export mesh, authoritative result

**Shared-edge GPU extraction**:
The Marching Cubes path that marks active grid edges on CUDA, assigns one vertex per active edge, and transfers only shared vertices plus final face indices instead of three duplicate vertices per triangle.
_Avoid_: GPU-only topology repair, duplicate-triangle vertex output

**Implicit surface view**:
An opaque rendering of the zero crossing of an implicit volume, without first producing a triangulated surface mesh.
_Avoid_: Density volume, translucent volume

**Interactive implicit view**:
A bounded-fidelity implicit surface view intended to remain responsive during camera and section-plane interaction.
_Avoid_: Precise render, export-quality view

**Precise implicit render**:
A separate high-fidelity image of the implicit surface used to inspect geometric truth when the interactive view is insufficient.
_Avoid_: Interactive preview

**Transition plane**:
A user-selected plane that divides the design domain into the two sides used to assign the G and D lattice fields.

**Signed plane distance**:
The physical distance from a sampled point to the transition plane; its sign identifies the side of the plane.

**Cell map**:
A regular hexahedral parameterization covering the design-domain bounding box. It defines the continuous TPMS field's physical cell spacing and sampling layout; it is not a collection of independently meshed STL blocks.

**Anisotropic cell map**:
A cell map with independently controlled physical spacing along X, Y, and Z. It represents a rectangular-period TPMS parameterization and must be treated separately from the standard isotropic cubic-cell case.

**Cell-map coordinate transform**:
The shared mapping from world coordinates to normalized cell coordinates. It owns cell spacing, map origin, and periodic indexing so each lattice family does not need to reimplement physical-to-local coordinate handling.

**Lattice family evaluator**:
The family-specific implicit or unit-cell evaluator that consumes normalized cell coordinates and user parameters. TPMS families and future non-TPMS unit cells share the cell-map pipeline while retaining their own field or geometry definition.

**Cell map view**:
An independent spatial preview of the cell map and design-domain relationship. It describes the generation layout without being a generated lattice result or a derived STL mesh.

**Cell map padding**:
The complete-cell extension between the design-domain bounding box and the cell map boundary. It preserves whole periodic cells before the design-domain field performs the final spatial restriction.

**Fitted cell map**:
A regular Cell Map whose outer bounds exactly match the design-domain bounding box. User-entered cell sizes are targets; uniform actual cell sizes are derived from the nearest positive integer cell counts on each axis.
_Avoid_: Conformal cell map, surface-following grid

**Target cell size**:
The user-requested periodic size used to choose fitted Cell Map counts. It may differ slightly from the actual cell size.

**Actual cell size**:
The uniform per-axis spacing represented by a Cell Map and consumed by its lattice-family evaluator.

**Implicit-field repair pass**:
A bounded modification of a sampled implicit field, such as closing a gap smaller than the user-selected tolerance, followed by a fresh derived-surface extraction and topology check. It is distinct from arbitrary triangle-mesh surgery.

**Export tolerance**:
The maximum geometric deviation accepted when deriving and simplifying a surface mesh for STL export. It controls sampling and simplification resolution indirectly; it is not a requested triangle count.

**Accelerated mesh simplifier**:
A compiled geometry backend used to reduce a derived surface mesh under an export-tolerance constraint. It must return to the original mesh when simplification worsens topology status.

**Export reconstruction**:
The fresh derived-surface extraction performed for STL export from the authoritative implicit evaluator and export tolerance. It is independent of the interactive display cache and its display voxel size.

**Unit-cell provider**:
The pluggable source of one lattice family's local field or periodic unit geometry. The shared cell-map, design-domain restriction, extraction, repair, validation, and export pipeline consumes this provider without depending on TPMS-specific formulas.

**Prepared custom unit cell**:
A watertight STL body plus a three-dimensional source Domain and a private resident mesh-distance backend. Source X/Y/Z map to Cell Map U/V/W; the source body owns its thickness and is evaluated periodically without copying triangles per cell.
_Avoid_: Parametric TPMS cell, duplicated cell meshes

**Custom unit-cell Domain**:
The source-space bounds defining one complete period of a prepared custom unit cell. The imported body's cleaned bounding box is the default Domain; changing Cell Map dimensions scales this Domain into one U/V/W cell.
_Avoid_: Design domain, Cell Map bounds

**Custom-cell target feature thickness**:
An optional physical thickness target applied after Cell Map scaling by a world-space signed-distance offset. Unset preserves the imported STL zero set. A requested thickness change moves exposed sides by half that change; intersections and highly curved regions are not guaranteed to have one exact local minimum thickness.
_Avoid_: Cell size, source-mesh scale, guaranteed local wall thickness

**Trimmed custom-cell extraction domain**:
The design-domain AABB used as the authoritative bounds of a custom-cell intersection. Its extraction grid remains anchored to the Cell Map so removing outside ranges does not alter voxel spacing or sampling phase.
_Avoid_: Reduced resolution, cropped design geometry

**Per-family cell map**:
The cell map associated with one lattice family. G and D may use different physical spacings; a transition combines their evaluators inside one common padded spatial envelope without forcing their periods to match.

**Cell-map frame**:
A right-handed local UVW coordinate system defined by an origin and independent U- and V-axis directions; W is derived from their cross product. In follow-origin mode the origin is the design domain's frame-aligned minimum corner and changes with U/V; in manual mode it is an explicit lattice phase anchor. The frame controls the position, phase, and orientation of a Cell Map, while cell sizes are measured along U, V, and W.

**Frame-aligned minimum origin**:
The world-space point whose U, V, and W coordinates are the independent minima of all design-domain points in the current Cell-map frame. It is the default Cell Map starting point when origin following is enabled.
_Avoid_: World-axis bounding-box minimum, Cell Map world AABB minimum

**Signed cell-index range**:
The integer U, V, and W index intervals required to cover the design domain relative to a Cell-map frame. Negative indices are valid when the design domain extends onto a negative side of the frame origin.

**Frame-aligned complete-cell coverage**:
The Cell Map coverage used for a custom Cell-map frame. It preserves the requested U, V, and W cell sizes and the frame's phase anchor, permits complete cells outside the design domain, and relies on the design-domain field for final trimming. Fitted bounds apply only to the default world-aligned frame.

**Per-family cell-map frame**:
The independently configured Cell-map frame owned by one lattice family. G and D retain separate frames, including when their fields are combined in a transition.

## Transition terms

**Field-driven transition region**:
A finite scalar-value interval of a selected spatial field that assigns the two transition operands and drives their Ramp weight; for signed-distance fields, negative values identify the inside and positive values identify the outside.
_Avoid_: Transition plane, geometry Boolean

**Transition driver field**:
The selected scalar field used to define a field-driven transition region independently of the two transition operands. A transition initially references one driver field, while the scene may retain multiple candidate fields.
_Avoid_: Transition operand, display-only field plane

**Field transition interval**:
The ordered lower and upper scalar values of a transition driver field at which the Ramp reaches its two exact endpoint operands. For a signed-distance driver its values are physical millimetres; its midpoint and span translate respectively to Ramp centre offset and transition width.
_Avoid_: Colour range, display-field bounds

**Implicit primitive**:
A named, parameterized analytic signed-distance field for a sphere, cylinder, or box. Each primitive has an independent transform and remains an editable scene object even when it is not selected as the transition driver field.
_Avoid_: STL primitive mesh, transition result

**Primitive transform**:
The independently editable world-space position and right-handed orientation of an implicit primitive. The selected primitive is moved along its displayed X/Y/Z axes and rotated around those axes through explicit rotation handles; a sphere retains this transform for interaction consistency although its SDF is orientation-invariant.
_Avoid_: Cell-map frame, transition-plane orientation

**Unclipped primitive preview**:
The complete visual representation of an implicit primitive, retained outside the design domain for placement and field inspection. It does not imply material exists outside the final design-domain-constrained transition result.
_Avoid_: Generated transition geometry, design-domain overlay

**Field-object scene**:
The Design document-private collection of independently named implicit
primitives available for placement, visual inspection, and later field-driven
operations. A transition driver references one scene object without taking
ownership of it. Reuse in another Design document creates an independent copy,
never a shared mutable reference.
_Avoid_: Transition configuration, generated lattice result

**Transition band**:
The region around the transition plane where both lattice fields contribute to the blended field.

**Sigmoid weight**:
A smooth value in the range [0, 1] that changes monotonically across the transition band and assigns the relative contribution of the two lattice fields.

**Transition weight policy**:
The base morphology-blending curve used before topology constraints are
enforced. Automatic selection is the default. Advanced users may explicitly
select linear (C0), smoothstep (C1), smootherstep (C2), sigmoid/tanh, or cosine
weighting. The spatial Ramp maps plane distance across a finite transition
width and clamps to exact endpoint values outside it. A weight curve does not
itself guarantee connectivity or minimum feature thickness. Transition width
is the sole normal-mode morphology control. Sigmoid sharpness is available only
in advanced mode when sigmoid/tanh weighting is explicitly selected.
_Avoid_: Transition topology operator, connectivity guarantee

**Blended field**:
The weighted combination of the G and D TPMS fields before intersection with the design-domain field.

**Topology-constrained transition**:
The target transition behavior in which cross-band material connectivity, the
configured minimum feature thickness, and the absence of isolated fragments
are hard constraints. The operator may add or remove a bounded amount of
material inside the transition band; exact scalar-field interpolation and
visual preservation of each source unit cell are secondary objectives. Outside
the user-configured transition band, each source lattice must remain identical
to its independently generated result; all transition topology edits are
strictly confined to the band. Connectivity acceptance requires a valid
material connection across the band. Whole-result component count is validated
and reported, but this operator does not modify out-of-band geometry merely to
force the complete result to one connected component.
_Avoid_: Pure weight blending, visual-only smoothing

**Transition minimum feature thickness**:
The independently configurable lower bound for material features created or
retained inside a topology-constrained transition band. Its automatic default
is the smaller characteristic thickness of the two operands. A custom unit
cell supplies that value through its configured or measured characteristic
thickness.
_Avoid_: STL repair tolerance, display voxel size, source wall thickness

**Recommended transition width**:
The non-binding lower width estimated from both operands' cell dimensions,
frame mismatch, and characteristic thickness. A narrower user value remains
valid input and is never silently replaced; the UI shows an inline risk state
and offers an explicit action to adopt the recommendation.
_Avoid_: Minimum allowed transition width, automatic width override

**Transition local registration**:
The in-band smooth adjustment of one or both operands' local Cell-map
coordinates to improve phase and feature alignment when cell size, orientation,
or phase differs. It is enabled automatically by default, remains confined to
the transition band, and converges exactly to each operand's original Cell Map
at the corresponding band boundary.
_Avoid_: Global Cell-map modification, source-frame replacement

**Transition operand**:
Either side of a transition, selected through the common Unit-cell provider
interface rather than a hard-coded lattice-family pair. G, D, and custom STL
providers are initial operands. Two operands may use the same provider type
with different parameters, Cell Maps, or Frames. Future providers participate
without pair-specific transition implementations.
_Avoid_: G-side, D-side, lattice-pair branch

**Transition execution backend**:
The capability-driven compute path selected for a pair of Transition operands.
GPU execution is preferred when both providers support it. Otherwise the
system falls back to batched CPU or a supported hybrid path, and reports the
actual backend and expected performance instead of rejecting the transition.
_Avoid_: Guaranteed GPU transition, hidden CPU fallback

## Quality terms

**Connectivity defect**:
An unintended separation of a generated lattice into multiple components or a broken local connection.

**Periodic seam compatibility**:
The property that a custom unit cell's solid cross-sections on each required pair of opposite U, V, and W Domain faces correspond under periodic identification closely enough to form material connections between neighbouring cells. It is an optional quality requirement because some authored unit cells intentionally remain disconnected in one or more periodic directions.
_Avoid_: Watertight source mesh, connected source component, display-island cleanup

**Periodic seam bridge**:
An optional local material extension at selected periodic Cell Map faces. It
uses the union of the two opposite-face solid sections and extends that section
by a bounded physical depth across the seam. The user explicitly selects U, V,
and/or W; unselected directions are only checked and warned about.
_Avoid_: Global cell thickening, automatic direction selection, topology repair

**Non-watertight result**:
A generated mesh with boundary edges, non-manifold topology, or another condition that prevents it from representing a closed solid volume.

**Topology status**:
The quality state of a derived surface mesh, including watertightness, winding consistency, and connected-component count. A warning status informs export decisions but does not by itself prevent STL export.

**Transition quality status**:
The non-blocking acceptance result for cross-band connectivity, transition
minimum feature thickness, and isolated-fragment checks. An unsuccessful
result remains available for inspection and export. The UI reports failed
checks and locations through a lightweight inline status and export metadata;
it does not open a warning dialog.
_Avoid_: Export blocker, modal transition warning

**Authoritative transition validation**:
The deterministic validation of transition constraints directly against the
authoritative implicit geometry within the transition band. It uses an
independent adaptive inspection resolution with local refinement in risky
regions; display quality and STL reconstruction tolerance do not change its
pass/fail result.
_Avoid_: Display-mesh validation, exported-STL-only validation

**Numerical fragment**:
A closed disconnected component whose volume is below both the selected export tolerance scale and a strict fraction of the main component. It is a sampling artifact, not an engineering-scale lattice component.
_Avoid_: Small component, disconnected lattice

**Sampled-field display island**:
A negative 26-neighbour component containing no more than six samples whose local reconstructed volume is below both one display-voxel volume and a strict fraction of the main occupancy proxy. It may be suppressed for stable visualization without modifying the authoritative implicit body or STL reconstruction.
_Avoid_: Authoritative fragment cleanup, lattice component deletion

**Repair result**:
A new derived surface mesh produced by an explicit topology-repair operation; it does not replace the original derived surface mesh until the user chooses which result to inspect or export.

## Computational-risk terms

**Large-grid memory issue**:
A generation request whose voxel grid and intermediate arrays exceed the
available process memory. The current mitigation is optional slab batching;
the long-term issue is tracked as TPMS-002.

## Layer inspection terms

**Layer contour view**:
The display-only inspection mode that samples one constant-coordinate plane
of a selected implicit result and renders its level-set polylines. It does not
generate triangles or modify the authoritative body.

**Contour field source**:
The selected representation used for layer inspection: the sampled implicit
render field, the STL-reconstruction sampling field, or the authoritative
evaluator-backed implicit body. The source is explicit so differences caused
by display sampling or export sampling remain observable.

**Layer contour cache**:
An LRU cache keyed by result identity, field source, layer axis and position,
level, and effective sampling spacing. It stores only 2-D contour results and
is invalidated when the design domain or generated result changes.

**Contour sampling spacing**:
The in-plane distance between samples used to extract a layer contour. Render
and authoritative inspection reuse the display sampling spacing; STL
inspection uses the effective STL reconstruction spacing, including the
current tolerance and spacing mode.
