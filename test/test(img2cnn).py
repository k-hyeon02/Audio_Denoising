from scipy.ndimage import median_filter
import numpy as np
import matplotlib.pyplot as plt

# img = np.load('/Users/hyeon/Desktop/경희대/2학년 2학기/dlearn/Audio_Denoising/test/image.npy', allow_pickle=True)
img = plt.imread('/Users/hyeon/Desktop/경희대/2학년 2학기/dlearn/Audio_Denoising/test/image.npy')

fig, axes = plt.subplots(1, 3, figsize=(15, 4))


np.save('test_spec.npy', img)

test_spec = np.load('test_spec.npy')


# A 모양으로 미디언 필터 (가로선만 남김 -> 보컬/악기 추출)
harmonic = median_filter(test_spec, footprint=np.ones((3, 1, 15)))
harmonic = 1 - harmonic

# B 모양으로 미디언 필터 (세로선만 남김 -> 드럼/노이즈 추출)
percussive = median_filter(test_spec, footprint=np.ones((3, 15, 1)))

axes[0].imshow(img, cmap="gray")
axes[0].set_title("original")
axes[1].imshow(harmonic, cmap="gray")
axes[1].set_title("harmonic")
axes[2].imshow(percussive, cmap="gray")
axes[2].set_title("percussive")

plt.tight_layout()
plt.show()