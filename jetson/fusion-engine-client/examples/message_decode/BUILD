package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "message_decode",
    srcs = [
        "message_decode.cc",
    ],
    data = [
        ":example_data",
    ],
    deps = [
        "//common:print_message",
        "@fusion_engine_client",
    ],
)

filegroup(
    name = "example_data",
    srcs = ["example_data.p1log"],
)
