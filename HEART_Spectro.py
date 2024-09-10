import RPi.GPIO as GPIO
import time, board, datetime
import pygame
from adafruit_as726x import AS726x_I2C

# ============= PARAMETERS ============
METHOD = 'SUM' 		# Choose between 'SUM' and 'AVERAGE' 
			# SUM adds light count over the period (greater definition), avarage averages it.

SPECTRA_PERIOD = 5	#in s for averaged_spectra, in count for sum_spectra

# Define leds and corresponding ports
LEDs={
	0:4,
	385:13,
	400:19,
	457:26,
	650:5}

# ==== Setup =====
# -- LEDs
GPIO.setmode(GPIO.BCM)
for _, led in LEDs.items():
	GPIO.setup(led, GPIO.OUT)

# -- Sensor
i2c = board.I2C()
sensor=AS726x_I2C(i2c)
sensor.conversion_mode=sensor.MODE_2
print(f"Temp: {sensor.temperature}")



def graph_map(x):
	return min(int(x* 80/16000),80)


def averaged_spectra(period=1, _delta=0.1):
	# By default will collect data every 100ms
	data={450:[], 500:[], 550:[], 575:[], 600:[], 650:[]}

	_start=time.time()
	deltaT=float(period-float(time.time()-_start)-_delta)

	while deltaT>0:
		try:
			if sensor.data_ready:
				mapping={
					450:sensor.violet,
					500:sensor.blue,
					550:sensor.green,
					575:sensor.yellow,
					600:sensor.orange,
					650:sensor.red}
				
				for nm, _sens in mapping.items():
					data[nm].append(_sens)
 
		except Exception as e:
			 print(e)
		 
		# Get remaining time
		time.sleep(_delta)
		deltaT=float(period-float(time.time()-_start)-_delta)
	# print(f"Averaged: {len(data[450])} spectra")
	# Average data
	for nm, lst in data.items():
		data[nm] = sum(data[nm])#/len(data[nm])
	
	return data
	
def sum_spectra(_count=4):
	# By default will collect data every 100ms
	data={450:[], 500:[], 550:[], 575:[], 600:[], 650:[]}
	c=0
	while c<=_count:
		try:
			while not sensor.data_ready:
				time.sleep(0.1)
			mapping={
					450:sensor.violet,
					500:sensor.blue,
					550:sensor.green,
					575:sensor.yellow,
					600:sensor.orange,
					650:sensor.red}
				
			for nm, _sens in mapping.items():
				data[nm].append(float(_sens))
 
		except Exception as e:
			 print(e)
		c+=1

	# Average data
	for nm, lst in data.items():
		data[nm] = sum(data[nm])
	
	return data    
	

def main():
    
    # Initiate pygame
    pygame.init()
    display = pygame.display.set_mode((300, 300))
    _quite=False
	
    # Start sensor
    time.sleep(0.5)
    _cycle=0
    filename=f"/home/jules/Downloads/Data/spectra_{datetime.datetime.now().strftime('%Y-%m-%d %H:%H:%M')}.csv"


    
    while True:
        for nm, led in LEDs.items():
            #print(f"{nm}nm ON")
            GPIO.output(led, GPIO.HIGH)

            # Wait for data to be ready
            while not sensor.data_ready:
                time.sleep(0.1)
            
            # get spectra 
	    if METHOD == 'AVERAGE':
            	data = average_spectra(_count=SPECTRA_PERIOD)
	    else:
		data = sum_spectra(_count=SPECTRA_PERIOD)
            data.update({'nm':nm})
            
            #Append to file
            # Open file to append
            with open(filename, 'a') as f:
                f.write(f"{str(data)},")
            
            #Display Data on terminal:
            print()
            print(f"Cycle: {_cycle}")
            for wl, v in data.items():
                print(f"{wl}: " + graph_map(v)*10*"=")
    
            GPIO.output(led, GPIO.LOW)
            
            
            #Check for "q" press
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        _quite=True
            if _quite is True:
                print("Exit")
                exit()
        _cycle+=1


	
	
if __name__ == "__main__":
	main()
	
