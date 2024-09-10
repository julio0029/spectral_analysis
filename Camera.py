import picamera
import datetime, time
import pickle
import keyboard
import pygame


def main():
    pygame.init()
 
    # creating display
    display = pygame.display.set_mode((300, 300))
   
    with picamera.PiCamera() as camera:
        camera.resolution=(640,480)
        camera.framerate=24
        camera.rotation=180
        camera.start_preview()
        filename=f'recording_{datetime.datetime.now().strftime("%Y-%m-%d %H:%H:%M")}.h264'
        camera.start_recording(filename, format='h264', quality=20)

        # press q or esc
        _quite=False
        while True:
            camera.wait_recording(10)
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        _quite=True

            if _quite==True:
                # Close preview and save file
                camera.stop_preview()
                camera.stop_recording()
                print("Saved video")

if __name__ == "__main__":
    main()