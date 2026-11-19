"""Genera una foto cenital sintética (campo de naranjos) para probar sin cámara."""
import numpy as np, cv2
rng = np.random.default_rng(3)
W, H = 1400, 1050
bg = np.zeros((H, W, 3), np.float32); bg[:] = (120, 175, 190)  # BGR tierra clara
n = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 25) * 60
bg += n[..., None]
img = np.clip(bg, 0, 255).astype(np.uint8)
cv2.rectangle(img, (0, 560), (W, 600), (205, 225, 235), -1)          # camino horizontal
cv2.rectangle(img, (700, 0), (730, H), (205, 225, 235), -1)           # camino vertical
xs = np.arange(W); riv = (645 + 18 * np.sin(xs / 190)).astype(int)
for x in xs: cv2.line(img, (x, riv[x] - 20), (x, riv[x] + 20), (160, 120, 60), 1)   # orilla
for x in xs: cv2.line(img, (x, riv[x] - 12), (x, riv[x] + 12), (190, 120, 40), 1)   # agua (BGR azul)
def orchard(x0, y0, x1, y1, step=34):
    for yy in range(y0 + step // 2, y1, step):
        for xx in range(x0 + step // 2, x1, step):
            j = rng.integers(-3, 4, 2); c = (xx + j[0], yy + j[1]); r = int(rng.integers(12, 15))
            cv2.circle(img, (c[0] + 6, c[1] + 6), r, (60, 80, 60), -1)                  # sombra
            cv2.circle(img, c, r, (40, int(rng.integers(95, 125)), 40), -1)             # copa oscura
            cv2.circle(img, (c[0] - 3, c[1] - 3), r // 2, (50, 140, 70), -1)           # luz
for box in [(80, 80, 640, 500), (780, 80, 1320, 500), (780, 680, 1320, 980)]: orchard(*box)
for (x, y, w, h, col) in [(110, 700, 70, 50, (60, 70, 190)), (230, 740, 55, 45, (50, 60, 170)), (400, 720, 50, 40, (225, 225, 225))]:
    cv2.rectangle(img, (x, y), (x + w, y + h), col, -1)
img = cv2.GaussianBlur(img, (0, 0), 1.0)
cv2.imwrite("sample_aerial.jpg", img)
print("sample_aerial.jpg")
