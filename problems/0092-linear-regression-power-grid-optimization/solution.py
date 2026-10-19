import math

PI = 3.14159

def power_grid_forecast(consumption_data):
	# 1) Subtract the daily fluctuation (10 * sin(2π * i / 10)) from each data point.
	# 2) Perform linear regression on the detrended data.
	# 3) Predict day 15's base consumption.
	# 4) Add the day 15 fluctuation back.
	# 5) Round, then add a 5% safety margin (rounded up).
	# 6) Return the final integer.
	detrended = []
	for i,y in enumerate(consumption_data):
		fluctation = 10 * math.sin(2*math.pi * i / 10)
		detrended.append(y - fluctation)
	n = len(detrended)
	x = list(range(1,n+1))
	mean_x = sum(x) / n 
	mean_y = sum(detrended) / n 

	num = sum((xi - mean_x) * (yi - mean_y)for xi,yi in zip(x, detrended))
	den = sum((xi - mean_x) ** 2 for xi in x)
	b = num / den
	a = mean_y - b * mean_x
	day_15_base = a + b * 15
	f_15 = 10 * math.sin(2*math.pi * 15 / 10)
	day_15_total = day_15_base + f_15
	safe_total = math.ceil(day_15_total *  1.05)
	return safe_total