from os import listdir
from statistics import mean, median
from os.path import isfile, join
from PIL import Image

path = '/source/wikiart/'
onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

# Filter files by extension
onlyfiles = [f for f in onlyfiles if f.endswith('.jpg')]

track_width = []
track_height = []
num_images = 0
for filename in onlyfiles:
    im = Image.open(path + filename)
    width, height = im.size
    track_width.append(width)
    track_height.append(height)
    num_images += 1

average_width = int(mean(track_width))
average_height = int(mean(track_height))
median_width = int(median(track_width))
median_height = int(median(track_height))
min_width = min(track_width)
min_height = min(track_height)
max_width = max(track_width)
max_height = max(track_height)

print("\nNumber of images: " + str(num_images))
print('\nAverage width: ' + str(average_width))
print('Average height: ' + str(average_height))
print('Median width: ' + str(median_width))
print('Median height: ' + str(median_height))
print('Min width: ' + str(min_width))
print('Min height: ' + str(min_height))
print('Max width: ' + str(max_width))
print('Max height: ' + str(max_height))


