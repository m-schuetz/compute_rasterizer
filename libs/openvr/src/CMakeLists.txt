
project(openvr_api)

IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	add_definitions(-DOSX -DPOSIX)
	set_source_files_properties(vrcommon/pathtools_public.cpp vrcommon/vrpathregistry_public.cpp PROPERTIES COMPILE_FLAGS "-x objective-c++")
ELSEIF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
	add_definitions(-DLINUX -DPOSIX)
ELSEIF(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
	add_definitions(-DWIN32)
	if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        add_definitions(-D_WIN64)
    endif()
ENDIF()


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
include_directories(. ../headers)
add_library(openvr_api STATIC openvr_api_public.cpp jsoncpp.cpp vrcommon/dirtools_public.cpp vrcommon/envvartools_public.cpp vrcommon/pathtools_public.cpp vrcommon/sharedlibtools_public.cpp vrcommon/hmderrors_public.cpp vrcommon/vrpathregistry_public.cpp vrcommon/strtools_public.cpp vrcommon/vrpathregistry_public.cpp)

install(TARGETS openvr_api DESTINATION lib)