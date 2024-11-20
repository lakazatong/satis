import cv2
import numpy as np

black = (0, 0, 0)
r_color = (255, 0, 0)
g_color = (0, 255, 0)
b_color = (0, 0, 255)

reference_colors = np.array([r_color, g_color, b_color])

def closest_color(r, g, b):
	if r < 10 and g < 10 and b < 10:
		return black
	global reference_colors
	color = np.array((r, g, b))
	distances = np.linalg.norm(reference_colors - color, axis=1)
	return tuple(reference_colors[np.argmin(distances)])

def extract_and_filter_colors(video_path, x, y):
	
	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		raise ValueError(f"Cannot open video file: {video_path}")
	previous_color = None
	while True:
		ret, frame = cap.read()
		# cv2.imwrite('test.png', frame)
		# break
		if not ret: break
		b, g, r = frame[y, x]
		current_color = closest_color(r, g, b)
		if current_color != previous_color:
			previous_color = current_color
			yield current_color
	cap.release()

video_path = "video.mp4"
 
# out = ''
out = 'rbgrbbrbbgrbbrbgrbbrbrbgbrbbrbgrbbrbgrbbrbrbgbrbbrbgrbbrbgrbbrbbgrbbrbgrbbrbgrbbrbrbgbrbbrbgrbbrbgrbbrbrbgbrbbrbgrbbrbgrbbrbrbgbrbbrbgrbbrbgrbbrbrbgbrbbrbgrbbrbgrbbrbrbgbrbbrbgrbbrbrgrbbrbrbgbrbbrbgrbbrbgrbbrbrbgbrbbrbgrbbrbgrbbrbrbgbrbbrbgrbbrbgrbbrbrbgbrbbrbgrbbrbgrbbrbrbgbrbbrbgrbbrbrgrbbrbbgrbbrbgrbbrbrbgbrbbrbgrbbrbgrbbrbbgrbbrbgrbbrbrbgbrbbrbgrbbrbgrbbrbrbgbrbbrbgrbbrbgrbbrbbrgbrbbrbgrbbrbgrbbrbbgrbbrbgrbbrbrbgbrbbrbgrbbrbgrbbrbrbgbrbbrbgrbbrbgrbbrbgrbbrbrbgbrbbrbgrbbrbgrbbrbrbgbrbbrbgrbbrbgrbbrbrbgbrbbrbgrbbrbgrbbrbrbgbrbbrbgrbbrbgrbbrbbrgbrbbrbgrbbrbgrbbrbrbgbrbbrbgrbbrbgrbbrbrbgbrbbrbgrbbrbgrbbrbrbgbrbbrbgrbbrbgrbbrbbrgbrbbrbgrbbrbgrbbrbrbgbrbbrbgrbbrbgrbbrbbgrbbrbgrbbrbrbgbrbbrbgrbbrbgrbbrbrbgbrbbrbgrbbrbgrbbrbrbgbrbbrbgrbbrbgrbbrbrbgbrbbrbgrbbrbgrbbrbrbgbr'
# print(out.startswith(new_out))
# print('\n'.join(out.split('g')))

# pattern = 'rbbrbrbgbrbbrbgrbbrbg'

print(len(out))

print(out.count('g'))
print(out.count('r'))
print(out.count('b'))

exit(0)

for color in extract_and_filter_colors(video_path, 900, 70):
	if color == black: continue
	if color == r_color:
		out += 'r'
	elif color == g_color:
		out += 'g'
	else:
		out += 'b'

print(out)
