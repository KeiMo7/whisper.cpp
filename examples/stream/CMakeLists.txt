# SDL/microphone independent streaming session library. Built unconditionally
# so offline tooling and the unit tests can use it without SDL2.
set(STREAM_SESSION_TARGET stream-session)
add_library(${STREAM_SESSION_TARGET} STATIC
    stream-session.h
    stream-session.cpp
    stream-whisper.h
    stream-whisper.cpp
    audio-ring.h
    )

include(DefaultTargetOptions)

target_include_directories(${STREAM_SESSION_TARGET} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/examples
    )
target_link_libraries(${STREAM_SESSION_TARGET} PUBLIC whisper)

# Offline, parseable alignment tool; no SDL2 dependency.
set(ALIGN_TARGET whisper-align)
add_executable(${ALIGN_TARGET} stream-align.cpp)

include(DefaultTargetOptions)

target_include_directories(${ALIGN_TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/examples
    )
target_link_libraries(${ALIGN_TARGET} PRIVATE common ${STREAM_SESSION_TARGET} whisper)

install(TARGETS ${ALIGN_TARGET} RUNTIME)

if (WHISPER_SDL2)
    set(TARGET whisper-stream)
    add_executable(${TARGET} stream.cpp)

    include(DefaultTargetOptions)

    target_include_directories(${TARGET} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/examples
        )
    target_link_libraries(${TARGET} PRIVATE
        common common-sdl ${STREAM_SESSION_TARGET} whisper ${CMAKE_THREAD_LIBS_INIT})

    install(TARGETS ${TARGET} RUNTIME)
endif ()
