
##############################################

configure_file(platform.h.in ${CMAKE_CURRENT_BINARY_DIR}/platform.h)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/layer)

set(ncnn_SRCS
    blob.cpp
    cpu.cpp
    gpu.cpp
    layer.cpp
    mat.cpp
    mat_pixel.cpp
    modelbin.cpp
    net.cpp
    opencv.cpp
    paramdict.cpp
    benchmark.cpp
)

macro(ncnn_add_layer class)
    string(TOLOWER ${class} name)

    # WITH_LAYER_xxx option
    if(${ARGC} EQUAL 2)
        option(WITH_LAYER_${name} "build with layer ${name}" ${ARGV1})
    else()
        option(WITH_LAYER_${name} "build with layer ${name}" ON)
    endif()

    message("WITH_LAYER_${name} = ${WITH_LAYER_${name}}")

    set(finalize_content "\n        int ret;")

    if(WITH_LAYER_${name})
        list(APPEND ncnn_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/layer/${name}.cpp")

        # look for arch specific implementation and append source
        # optimized implementation for armv7 aarch64
        if((ANDROID AND ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "armv7-a"))
            OR (ANDROID AND ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64"))
            OR (UNIX AND ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "armv7l"))
            OR (UNIX AND ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64"))
            OR (IOS AND ("${CMAKE_OSX_ARCHITECTURES}" STREQUAL "armv7"))
            OR (IOS AND ("${CMAKE_OSX_ARCHITECTURES}" STREQUAL "arm64"))
            OR (IOS AND ("${CMAKE_OSX_ARCHITECTURES}" STREQUAL "armv7;arm64")))
            if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/layer/arm/${name}_arm.cpp")
                list(APPEND ncnn_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/layer/arm/${name}_arm.cpp")

                file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/layer_final.h "#include \"layer/arm/${name}_arm.h\"\n")
                file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/layer_final.cpp "class ${class}_final : public ${class}_arm")

                set(finalize_content "${finalize_content}\n        ret = ${class}_arm::finalize();")
                set(finalize_content "${finalize_content}\n        if (ret != 0) return ret;")
            else()
                # naive C layer
                file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/layer_final.h "#include \"layer/${name}.h\"\n")
                file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/layer_final.cpp "class ${class}_final : public ${class}")

                set(finalize_content "${finalize_content}\n        ret = ${class}::finalize();")
                set(finalize_content "${finalize_content}\n        if (ret != 0) return ret;")
            endif()
        else()
            if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/layer/x86/${name}_x86.cpp")
                list(APPEND ncnn_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/layer/x86/${name}_x86.cpp")

                file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/layer_final.h "#include \"layer/x86/${name}_x86.h\"\n")
                file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/layer_final.cpp "class ${class}_final : public ${class}_x86")

                set(finalize_content "${finalize_content}\n        ret = ${class}_x86::finalize();")
                set(finalize_content "${finalize_content}\n        if (ret != 0) return ret;")
            else()
                # naive C layer
                file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/layer_final.h "#include \"layer/${name}.h\"\n")
                file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/layer_final.cpp "class ${class}_final : public ${class}")

                set(finalize_content "${finalize_content}\n        ret = ${class}::finalize();")
                set(finalize_content "${finalize_content}\n        if (ret != 0) return ret;")
            endif()
        endif()

        if(NCNN_OPENCL)
            # NOTE opencl
            if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/layer/opencl/${name}_opencl.cpp")
                list(APPEND ncnn_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/layer/opencl/${name}_opencl.cpp)

                file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/layer_final.h "#include \"layer/opencl/${name}_opencl.h\"\n")
                file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/layer_final.cpp ", public ${class}_opencl")

                set(finalize_content "${finalize_content}\n        ret = ${class}_opencl::finalize();")
                set(finalize_content "${finalize_content}\n        if (ret != 0) return ret;")

                # append opencl kernel source
                file(READ ${CMAKE_CURRENT_SOURCE_DIR}/layer/opencl/${name}.cl clcontent)
                string(REPLACE "\"" "\\\"" clcontent "${clcontent}")
                string(REPLACE "\n" "\\n\"\n\"" clcontent "${clcontent}")
                file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/layer_opencl_all.cl.h "\"")
                file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/layer_opencl_all.cl.h "${clcontent}")
                file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/layer_opencl_all.cl.h "\"\n")
            endif()
        endif()

        file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/layer_final.cpp "\n")
        file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/layer_final.cpp "{\n")
        file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/layer_final.cpp "public:\n")
        file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/layer_final.cpp "    virtual int finalize() {${finalize_content}\n")
        file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/layer_final.cpp "        return 0;\n")
        file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/layer_final.cpp "    }\n")
        file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/layer_final.cpp "};\n")
        file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/layer_final.cpp "::ncnn::Layer* ${class}_layer_creator() { return new ${class}_final; }\n")
        file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/layer_final.cpp "\n")

    endif()

    # generate layer_declaration and layer_registry file
    if(WITH_LAYER_${name})
        file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/layer_registry.h
            "#if NCNN_STRING\n{\"${class}\",${class}_layer_creator},\n#else\n{${class}_layer_creator},\n#endif\n")
    else()
        file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/layer_registry.h "#if NCNN_STRING\n{\"${class}\",0},\n#else\n{0},\n#endif\n")
    endif()

    # generate layer_type_enum file
    file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/layer_type_enum.h "${class} = ${__LAYER_TYPE_ENUM_INDEX},\n")
    math(EXPR __LAYER_TYPE_ENUM_INDEX "${__LAYER_TYPE_ENUM_INDEX}+1")
endmacro()

# create new
file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/layer_final.h)
file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/layer_final.cpp)
file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/layer_registry.h)
file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/layer_type_enum.h)
file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/layer_opencl_all.cl.h)
set(__LAYER_TYPE_ENUM_INDEX 0)

# layer implementation
ncnn_add_layer(AbsVal)
ncnn_add_layer(ArgMax OFF)
ncnn_add_layer(BatchNorm)
ncnn_add_layer(Bias)
ncnn_add_layer(BNLL)
ncnn_add_layer(Concat)
ncnn_add_layer(Convolution)
ncnn_add_layer(Crop)
ncnn_add_layer(Deconvolution)
ncnn_add_layer(Dropout)
ncnn_add_layer(Eltwise)
ncnn_add_layer(ELU)
ncnn_add_layer(Embed)
ncnn_add_layer(Exp)
ncnn_add_layer(Flatten)
ncnn_add_layer(InnerProduct)
ncnn_add_layer(Input)
ncnn_add_layer(Log)
ncnn_add_layer(LRN)
ncnn_add_layer(MemoryData)
ncnn_add_layer(MVN)
ncnn_add_layer(Pooling)
ncnn_add_layer(Power)
ncnn_add_layer(PReLU)
ncnn_add_layer(Proposal)
ncnn_add_layer(Reduction)
ncnn_add_layer(ReLU)
ncnn_add_layer(Reshape)
ncnn_add_layer(ROIPooling)
ncnn_add_layer(Scale)
ncnn_add_layer(Sigmoid)
ncnn_add_layer(Slice)
ncnn_add_layer(Softmax)
ncnn_add_layer(Split)
ncnn_add_layer(SPP OFF)
ncnn_add_layer(TanH)
ncnn_add_layer(Threshold)
ncnn_add_layer(Tile OFF)
ncnn_add_layer(RNN OFF)
ncnn_add_layer(LSTM OFF)
ncnn_add_layer(BinaryOp)
ncnn_add_layer(UnaryOp)
ncnn_add_layer(ConvolutionDepthWise)
ncnn_add_layer(Padding)
ncnn_add_layer(Squeeze)
ncnn_add_layer(ExpandDims)
ncnn_add_layer(Normalize)
ncnn_add_layer(Permute)
ncnn_add_layer(PriorBox)
ncnn_add_layer(DetectionOutput)
ncnn_add_layer(Interp)
ncnn_add_layer(DeconvolutionDepthWise)
ncnn_add_layer(ShuffleChannel)
ncnn_add_layer(InstanceNorm)
ncnn_add_layer(Clip)

add_library(ncnn STATIC ${ncnn_SRCS})

install(TARGETS ncnn ARCHIVE DESTINATION lib)
install(FILES
    blob.h
    cpu.h
    layer.h
    layer_type.h
    mat.h
    modelbin.h
    net.h
    opencv.h
    paramdict.h
    benchmark.h
    ${CMAKE_CURRENT_BINARY_DIR}/layer_type_enum.h
    ${CMAKE_CURRENT_BINARY_DIR}/platform.h
    DESTINATION include
)
