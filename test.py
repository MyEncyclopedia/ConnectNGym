import sys

import pygame
pygame.init()
display = pygame.display.set_mode((800,600))
clock = pygame.time.Clock()

while True:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			sys.exit(0)
		else:
			pygame.display.update()
			clock.tick(1)