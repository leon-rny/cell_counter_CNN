import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import czifile
from roifile import ImagejRoi

def onclick(event):
    if event.xdata and event.ydata:
        x, y = int(event.xdata), int(event.ydata)
        points.append((x, y))
        ax.plot(x, y, 'ro')
        fig.canvas.draw()

# Load the CZI image and ROI
cwd = os.getcwd()
number = '01_3'
image_path = cwd + f'/data/duodenum_{number}.czi'
roi_path = cwd + f'/data/duodenum_{number}.roi'
output_csv = f'data/cell_points_{number}.csv'

with czifile.CziFile(image_path) as czi:
    image = czi.asarray()
image = np.squeeze(image)

roi = ImagejRoi.fromfile(roi_path)
roi_coords = roi.coordinates()
closed_coords = np.vstack([roi_coords, roi_coords[0]])

channel_2 = image[1] * 25
normalized = (channel_2 - np.min(channel_2)) / (np.max(channel_2) - np.min(channel_2))

fig, ax = plt.subplots()
ax.imshow(normalized, cmap='Reds')
ax.plot(closed_coords[:, 0], closed_coords[:, 1], 'r--')
ax.set_title("Click on the image to select points")

points = []
cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()

if points:
    df = pd.DataFrame(points, columns=['x', 'y'])
    df.to_csv(output_csv, index=False)
    print(f"{len(points)} points saved as: {output_csv}")
else:
    print("No points were selected.")
