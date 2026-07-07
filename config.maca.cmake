# MetaX MACA backend configuration entry point for CMakeLists.txt
# SDK detection and compiler flags live in cmake/backends/maca.cmake

set(YIRAGE_USE_MACA ON CACHE BOOL "Enable MetaX MACA backend" FORCE)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/backends/maca.cmake)
