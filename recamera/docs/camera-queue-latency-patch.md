# Camera Queue Latency Patch

## Problem

The camera path inside `sscma-micro` was buffering too many frames before inference.

In:

- `components/sscma-micro/porting/sophgo/sg200x/ma_camera_sg200x.cpp`

each enabled camera channel created a queue like this:

```cpp
m_channels[i].queue = new MessageBox(param.fps);
```

That means the queue depth was roughly equal to the camera FPS.

At runtime, frame callbacks continuously posted captured frames into that queue, and `retrieveFrame()` later fetched them for inference. If inference was slower than capture, stale frames accumulated and the system processed older frames instead of the newest one.

This caused visible delay between what the camera currently saw and what the model was actually processing.

## Why queue size 0 is not the fix

`MessageBox` is a real queue abstraction and expects a positive size:

- `MessageBox(size_t size = 1)`

So a queue size of `0` is not a valid or safe solution here.

## Applied fix

The patch changes the queue behavior to **latest frame wins**:

1. Queue depth is set to `1`
2. If a new frame arrives while the queue is full:
   - the old queued frame is removed
   - that old frame is freed correctly
   - the newest frame is inserted instead

This keeps only one pending frame per channel and prevents backlog from growing.

## Result

Before:

- camera queue depth ~= FPS
- old frames could stack up
- inference lagged behind live video

After:

- camera queue depth = `1`
- stale frames are discarded
- inference receives the most recent available frame

## File changed

- [components/sscma-micro/porting/sophgo/sg200x/ma_camera_sg200x.cpp](/home/loris/Delivery/School/robocar/recamera/components/sscma-micro/porting/sophgo/sg200x/ma_camera_sg200x.cpp)

## Important note

If latency is still visible after this patch, also check the application loop delay in `src/main.cpp`:

- `DEBUG_TICK_DELAY_MS`

That delay is separate from the camera queue and can add noticeable lag on its own.
