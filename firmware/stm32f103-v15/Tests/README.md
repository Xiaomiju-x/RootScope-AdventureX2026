# 主机侧逻辑测试

这里的 mock 只验证应用逻辑，不代替 Keil 编译和实物测试。

使用 GCC：

```powershell
gcc -std=c11 -Wall -Wextra -Werror `
  -ITests/mock -ICore/Inc `
  Core/Src/stepper.c Core/Src/app.c Tests/test_app.c `
  -o Tests/test_app.exe
.\Tests\test_app.exe
```
