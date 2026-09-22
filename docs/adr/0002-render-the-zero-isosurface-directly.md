# Render the zero isosurface directly

The default implicit-volume view renders the zero isosurface as an opaque, lit surface using GPU ray casting. Traditional translucent density rendering is excluded from the initial scope because it does not communicate TPMS solid geometry clearly, while extracting a temporary triangle mesh would preserve the performance problem this feature is intended to remove.
