# SSCMA-Micro Missing Dependency Error

## Symptom

During CMake configure, the build fails with errors like:

```text
CMake Error at components/sscma-micro/CMakeLists.txt:4 (include):
  include could not find requested file:

    .../components/sscma-micro/sscma-micro/3rdparty/json/CMakeLists.txt

CMake Error at components/sscma-micro/CMakeLists.txt:6 (include):
  include could not find requested file:

    .../components/sscma-micro/sscma-micro/3rdparty/eigen/CMakeLists.txt
```

You may also see:

```text
fatal error: sscma.h: No such file or directory
```

## Cause

`components/sscma-micro/sscma-micro` is not a plain source folder in the original upstream layout. It is expected to contain the upstream `SSCMA-Micro` repository.

If that nested directory is empty or incomplete, the following files will be missing:

- `sscma-micro/sscma/sscma.h`
- `sscma-micro/3rdparty/json/CMakeLists.txt`
- `sscma-micro/3rdparty/eigen/CMakeLists.txt`

The sibling `sscma-example-sg200x` repository confirms this layout in its `.gitmodules` file:

```ini
[submodule "components/sscma-micro/sscma-micro"]
	path = components/sscma-micro/sscma-micro
	url = https://github.com/Seeed-Studio/SSCMA-Micro
```

## Fix

Restore the missing nested `SSCMA-Micro` repository under:

```text
components/sscma-micro/sscma-micro
```

One working approach is:

```bash
git clone https://github.com/Seeed-Studio/SSCMA-Micro components/sscma-micro/sscma-micro
```

Then rerun CMake:

```bash
cmake -S . -B build
```

During configure, `SSCMA-Micro` will fetch and populate:

- `3rdparty/json/cJSON`
- `3rdparty/eigen/eigen`

After that, build normally:

```bash
cmake --build build
```

## Notes

- This is not just a bad include path. The dependency payload is genuinely missing when the nested `sscma-micro` directory is empty.
- Network access may be required on the first configure because the vendor scripts fetch `cJSON` and `eigen`.
- If you want a stricter upstream-style setup, restore `components/sscma-micro/sscma-micro` as a git submodule instead of a plain cloned directory.
