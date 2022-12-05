############### Add include ###################
list(APPEND ADD_INCLUDE "include"
    )
list(APPEND ADD_PRIVATE_INCLUDE "")
###############################################

############ Add source files #################
# list(APPEND ADD_SRCS  "src/main.c"
#                       "src/scrfd_decode_cv.cpp"
#     )
# aux_source_directory(src ADD_SRCS)
append_srcs_dir(ADD_SRCS "src")
# list(REMOVE_ITEM COMPONENT_SRCS "src/test2.c")
###############################################

###### Add required/dependent components ######
list(APPEND ADD_REQUIREMENTS maix_cv_image libmaix)
###############################################

############ Add static libs ##################
# list(APPEND ADD_STATIC_LIB "lib/libtest.a")
list(APPEND ADD_STATIC_LIB "libaipudrv.a")
list(APPEND ADD_STATIC_LIB "libscrfd_decode.a")
###############################################

#### Add compile option for this component ####
#### Just for this component, won't affect other
#### modules, including component that depend
#### on this component
# list(APPEND ADD_DEFINITIONS_PRIVATE -DAAAAA=1)

#### Add compile option for this component
#### and components denpend on this component
# list(APPEND ADD_DEFINITIONS -DAAAAA222=1
#                             -DAAAAA333=1)
###############################################


set(CMAKE_C_LINK_FLAGS "${CMAKE_C_LINK_FLAGS} -Wl,--start-group /mnt/c/Users/btx00/Desktop/r329t/MaixPy3/ext_modules/libmaix/examples/nn_r329_scrfd_new/main/libscrfd_decode.a /mnt/c/Users/btx00/Desktop/r329t/MaixPy3/ext_modules/libmaix/examples/nn_r329_scrfd_new/main/libaipudrv.a -lm -lpthread -lstdc++ -Wl,--end-group -g" PARENT_SCOPE) 
register_component()

