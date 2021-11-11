cmake_minimum_required(VERSION 3.11)
project(Glitter)

option(GLFW_BUILD_DOCS OFF)
option(GLFW_BUILD_EXAMPLES OFF)
option(GLFW_BUILD_TESTS OFF)
add_subdirectory(Glitter/Vendor/glfw)

option(ASSIMP_BUILD_ASSIMP_TOOLS OFF)
option(ASSIMP_BUILD_SAMPLES OFF)
option(ASSIMP_BUILD_TESTS OFF)
add_subdirectory(Glitter/Vendor/assimp)

if(MSVC AND CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3 /std:c++17 /fp:fast")
else()
	message( FATAL_ERROR "Only 64bit MSVC is supported" )
endif()

include_directories(Glitter/Headers/
                    Glitter/Vendor/assimp/include/
                    Glitter/Vendor/glad/include/
                    Glitter/Vendor/glfw/include/
                    Glitter/Vendor/glm/
                    Glitter/Vendor/stb/
					Glitter/Vendor/fftw-x64/include/
					Glitter/Vendor/openvr/include/
					Glitter/Vendor/SRanipal/include/
					Glitter/Vendor/ZeroMQ/ZeroMQ/
					Glitter/Vendor/ZeroMQ/cppzmq/
					Glitter/Vendor/Eigen/)
					
find_library(FFTW libfftw3f-3 Glitter/Vendor/fftw-x64/lib)
find_library(OPENVR openvr_api Glitter/Vendor/openvr/lib)
find_library(SRANIPAL SRanipal Glitter/Vendor/SRanipal/lib)
find_library(ZEROMQ libzmq-v142-mt-4_3_3 Glitter/Vendor/ZeroMQ/lib)

include(FetchContent)

FetchContent_Declare(json
  GIT_REPOSITORY https://github.com/ArthurSonzogni/nlohmann_json_cmake_fetchcontent
  GIT_TAG v3.9.1)

FetchContent_GetProperties(json)
if(NOT json_POPULATED)
  FetchContent_Populate(json)
  add_subdirectory(${json_SOURCE_DIR} ${json_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

file(GLOB VENDORS_SOURCES Glitter/Vendor/glad/src/glad.c)
file(GLOB PROJECT_HEADERS Glitter/Headers/*.hpp
                          Glitter/Headers/*.h)
file(GLOB PROJECT_SOURCES Glitter/Sources/*.cpp)
file(GLOB PROJECT_SHADERS Glitter/Shaders/*)
file(GLOB PROJECT_CONFIGS CMakeLists.txt
                          Readme.md
                         .gitattributes
                         .gitignore
                         .gitmodules)

source_group("Headers" FILES ${PROJECT_HEADERS})
source_group("Shaders" FILES ${PROJECT_SHADERS})
source_group("Sources" FILES ${PROJECT_SOURCES})
source_group("Vendors" FILES ${VENDORS_SOURCES})

add_definitions(-DGLFW_INCLUDE_NONE
                -DPROJECT_SOURCE_DIR=\"${PROJECT_SOURCE_DIR}\")
add_executable(${PROJECT_NAME} ${PROJECT_SOURCES} ${PROJECT_HEADERS}
                               ${PROJECT_SHADERS} ${PROJECT_CONFIGS}
                               ${VENDORS_SOURCES})
target_link_libraries(${PROJECT_NAME} assimp glfw
                      ${GLFW_LIBRARIES} ${GLAD_LIBRARIES} ${FFTW} ${OPENVR} ${SRANIPAL} ${ZEROMQ}
                      )
					  
target_link_libraries(${PROJECT_NAME} nlohmann_json)
					  
set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${PROJECT_NAME})
	
add_custom_command(
    TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/Glitter/Vendor/fftw-x64/bin $<TARGET_FILE_DIR:${PROJECT_NAME}>
	COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/Glitter/Vendor/openvr/bin $<TARGET_FILE_DIR:${PROJECT_NAME}>
	COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/Glitter/Vendor/SRanipal/bin $<TARGET_FILE_DIR:${PROJECT_NAME}>
	COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/Glitter/Vendor/ZeroMQ/bin $<TARGET_FILE_DIR:${PROJECT_NAME}>
    DEPENDS ${PROJECT_SHADERS})
