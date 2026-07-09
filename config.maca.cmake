# MetaX MACA backend configuration entry point for CMakeLists.txt
# SDK detection and compiler flags live in cmake/backends/maca.cmake

set(YIRAGE_USE_MACA ON CACHE BOOL "Enable MetaX MACA backend" FORCE)
# Native .maca fingerprint kernels need fp16 header alignment (Loop R2);
# default OFF so transpiler + mxcc graph compile path can install on C500.
option(MACA_COMPILE_KERNELS "Compile native .maca fingerprint kernels with mxcc" OFF)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/backends/maca.cmake)
