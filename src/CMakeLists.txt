##
#  CMake script
##

SET(TARGET "wbm")
#SET(LIB_SRC model.cc)# bindings.cc)
SET(EXE_SRC ascii_utils.cc field_initializer.cc postprocess_vis.cc model.cc parameters.cc main.cc)

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)

PROJECT(${TARGET})

#FIND_PACKAGE(PythonLibs 3 REQUIRED)
#FIND_PACKAGE(Boost REQUIRED COMPONENTS python3)

FIND_PACKAGE(deal.II 9.0.0 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR})

IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

MESSAGE(STATUS "Found deal.II version ${DEAL_II_PACKAGE_VERSION} at '${deal.II_DIR}'")

DEAL_II_INITIALIZE_CACHED_VARIABLES()

# ADD_LIBRARY(lib${TARGET} SHARED ${LIB_SRC})
# DEAL_II_SETUP_TARGET(lib${TARGET})

ADD_EXECUTABLE(${TARGET} ${EXE_SRC})
#TARGET_LINK_LIBRARIES(${TARGET} lib${TARGET})
DEAL_II_SETUP_TARGET(${TARGET})

# SET_PROPERTY(TARGET lib${TARGET} APPEND PROPERTY
#     INCLUDE_DIRECTORIES "${PYTHON_INCLUDE_DIRS}")
# SET_PROPERTY(TARGET lib${TARGET} APPEND PROPERTY
#     INCLUDE_DIRECTORIES "${BOOST_INCLUDE_DIRS}")
# SET_PROPERTY(TARGET lib${TARGET} APPEND PROPERTY
#     LINK_LIBRARIES "${Boost_LIBRARIES}")
# SET_TARGET_PROPERTIES(lib${TARGET} PROPERTIES PREFIX "")

ADD_CUSTOM_TARGET(distclean
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target clean
    COMMAND ${CMAKE_COMMAND} -E remove_directory CMakeFiles
    COMMAND ${CMAKE_COMMAND} -E remove
        CMakeCache.txt cmake_install.cmake Makefile ${TARGET}.so
    COMMENT "distclean invoked"
    )

# Define custom targets to easily switch the build type:
ADD_CUSTOM_TARGET(debug
    COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
    COMMENT "Switch CMAKE_BUILD_TYPE to Debug"
    )

ADD_CUSTOM_TARGET(release
    COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
    COMMENT "Switch CMAKE_BUILD_TYPE to Release"
    )
