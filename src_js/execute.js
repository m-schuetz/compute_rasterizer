

// window.x = window.monitorWidth / 2;
// window.y = 0;
// window.width = window.monitorWidth / 2;
// window.height = window.monitorHeight;

window.width = 1920;
window.height = 1080;
window.x  = 2560 + 10;
window.y = 10;

log(view.position);

camera.near = 0.1;
camera.far = 10000;

CLOD_RANGE = [0.2, 0.2];
POINT_BUDGET_RANGE = [500 * 1000, 4 * 1000 * 1000];
USER_STUDY_TW = 0.001;

CLOD_BATCH_SIZE = 20 * 1000 * 1000;
LOD_UPDATES_ENABLED = false;
LOD_UPDATES_ENABLED = true;

USER_STUDY_RENDER_OCTREE = false;
USER_STUDY_RENDER_CLOD = true;

USER_STUDY_BLENDING = false;
EDL_ENABLED = false;
RENDER_DEFAULT_ENABLED = true;

USER_STUDY_OCTREE_MODE = "ADAPTIVE";
USER_STUDY_OCTREE_POINT_SIZE = 4;



USER_STUDY_LOD_MODIFIER = 0;

MSAA_SAMPLES = 1;

reportState(false);

// loadLAS("D:/dev/pointclouds/archpro/heidentor.las");