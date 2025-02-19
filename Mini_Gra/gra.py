import pygame
import sys
import math

pygame.init()

start_screen = pygame.display.set_mode((400, 300))
pygame.display.set_caption("Myszka za obiektem - Start")
start_font = pygame.font.Font(None, 52)
start_text = start_font.render("Naciśnij Start", True, (0, 0, 0))
start_text_rect = start_text.get_rect(center=(400, 200))  #Centrowanie napisu na ekranie

#Ustawienia okna gry
width, height = 800, 600
fullscreen = False
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Myszka za obiektem")

#Ustawienia obrazu tła (mapa)
background_image = pygame.image.load("mapa.PNG")
background_image = pygame.transform.scale(background_image, (width, height))

#Ustawienia obrazu prostokąta (obiektu)
flag_image = pygame.image.load("flaga.PNG")
flag_width, flag_height = 50, 30
flag_image = pygame.transform.scale(flag_image, (flag_width, flag_height))

rect_speed = 2
base_rect_speed = rect_speed
acceleration = 0.1  #Przyspieszenie

#Ustawienia obrazu myszki
mouse_image = pygame.image.load("flaga2.PNG")
mouse_size = 20
mouse_image = pygame.transform.scale(mouse_image, (mouse_size, mouse_size))

#Początkowe położenie prostokąta (zawsze w środku ekranu)
rect_x, rect_y = width // 2 - flag_width // 2, height // 2 - flag_height // 2

#Kolizja
def check_collision(rect_x, rect_y, rect_width, rect_height, mouse_x, mouse_y, mouse_size):
    distance = math.sqrt((rect_x + rect_width // 2 - mouse_x)**2 + (rect_y + rect_height // 2 - mouse_y)**2)
    return distance < (rect_width + mouse_size) / 2

def draw_start_screen(best_time):
    start_screen.fill((255, 255, 255))
    start_screen.blit(start_text, start_text_rect)

    if best_time is not None:
        best_time_text = start_font.render(f"Ostatni czas: {format_time(best_time)}", True, (0, 0, 0))
        start_screen.blit(best_time_text, (width // 2 - 160, 250))

#Formatowanie czasu
def format_time(milliseconds):
    seconds = milliseconds // 1000
    minutes = seconds // 60
    seconds %= 60
    return f"{minutes:02}:{seconds:02}"

# Sprawdź, czy kliknięcie nastąpiło w obszarze przycisku "Start"
def is_click_on_start_button(mouse_pos):
    return start_text_rect.collidepoint(mouse_pos)

game_active = False
start_time = 0
best_time = None 
last_acceleration_time = 0

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN and not game_active:
            mouse_pos = pygame.mouse.get_pos()
            if is_click_on_start_button(mouse_pos):
                game_active = True
                start_time = pygame.time.get_ticks() 
                rect_x, rect_y = width // 2 - flag_width // 2, height // 2 - flag_height // 2 
                rect_speed = base_rect_speed
                last_acceleration_time = start_time
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_f:
                fullscreen = not fullscreen
                if fullscreen:
                    screen = pygame.display.set_mode((width, height), pygame.FULLSCREEN)
                else:
                    screen = pygame.display.set_mode((width, height))
            elif event.key == pygame.K_m and game_active:
                pygame.display.toggle_fullscreen()

    if game_active:
        mouse_x, mouse_y = pygame.mouse.get_pos()

        #Sprawdź kolizję
        if check_collision(rect_x, rect_y, flag_width, flag_height, mouse_x, mouse_y, mouse_size):
            best_time = pygame.time.get_ticks() - start_time
            game_active = False
            rect_speed = base_rect_speed

        rect_dx = mouse_x - (rect_x + flag_width // 2)
        rect_dy = mouse_y - (rect_y + flag_height // 2)
        rect_distance = (rect_dx**2 + rect_dy**2)**0.5

        if rect_distance > rect_speed:
            rect_x += rect_speed * rect_dx / rect_distance
            rect_y += rect_speed * rect_dy / rect_distance
        else:
            rect_x, rect_y = mouse_x - flag_width // 2, mouse_y - flag_height // 2

        current_time = pygame.time.get_ticks()
        if current_time - last_acceleration_time >= 1000:
            rect_speed += acceleration
            last_acceleration_time = current_time

        screen.blit(background_image, (0, 0))
        screen.blit(flag_image, (rect_x, rect_y))
        screen.blit(mouse_image, (mouse_x - mouse_size // 2, mouse_y - mouse_size // 2))

        elapsed_time = pygame.time.get_ticks() - start_time
        elapsed_text = start_font.render(format_time(elapsed_time), True, (0, 0, 0))
        screen.blit(elapsed_text, (width - 100, 10))

        #Odśwież ekran
        pygame.display.flip()
    else:
        #Rysuj ekran startowy
        draw_start_screen(best_time)
        pygame.display.flip()
        pygame.time.Clock().tick(60)
