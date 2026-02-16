# Running FlashAttention in Xcode

To leverage Apple's powerful GPU debugging tools, follow these steps to set up your project in Xcode.

### 1. Create the Project
1. Open **Xcode**.
2. Select **File > New > Project**.
3. Choose **macOS** as the platform and **Command Line Tool** as the template.
4. Click **Next**, name it `FlashAttention`, and set the Language to **Objective-C** (don't worry, we will use your `.mm` file).

### 2. Add Your Files
1. Right-click the yellow folder named `FlashAttention` in the left sidebar.
2. Select **Add Files to "FlashAttention"...**
3. Select your `main.mm` and `kernels.metal`.
    * **CRITICAL:** If Xcode created a default `main.m`, delete it.
4. If prompted, click **Create Bridging Header** (though for this project it isn't strictly necessary).

### 3. Link Frameworks
Xcode needs to know you are using Metal.
1. Click the blue **FlashAttention** project icon at the very top of the sidebar.
2. Select the **FlashAttention** target.
3. Go to the **Build Phases** tab.
4. Expand **Link Binary With Libraries**.
5. Click the **+** button and search for/add:
    * `Metal.framework`
    * `Foundation.framework`
    * `CoreGraphics.framework`

### 4. Set the Working Directory (The "Secret" Step)
Your `main.mm` (Line 47) tries to load `kernels.metal` from the current folder. By default, Xcode runs apps in a hidden "DerivedData" folder where your file doesn't exist.
1. In the top bar, click **FlashAttention > Edit Scheme...**
2. Select **Run** on the left.
3. Go to the **Options** tab.
4. Check **Use custom working directory**.
5. Click the folder icon and select your actual project folder (`/Users/anewrag/Documents/HPC/Metal projects/flash_attention`).

### 5. Run & Debug
1. Press the big **Play** button (Cmd + R).
2. Once it's running, look at the bottom debug bar.
3. Click the **Small "M" Icon** (Metal icon) to capture a GPU frame.
4. You can now see every thread, variable, and buffer on the GPU!

### 6. Profiling (Optional)
If you want to see the performance charts:
1. Long-press the Play button and select **Profile** (or Cmd + I).
2. Choose **Metal System Trace**.
3. Press the red record button to see exactly how long your kernels take on the GPU timeline.
