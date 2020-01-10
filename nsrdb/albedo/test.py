import albedo as a
from datetime import datetime as dt
from datetime import timedelta
# year = ModisYear(2012, 'scratch')
dates = [(2015, 1)]  # , (2015, 9), (2013, 145)]
for y, d in dates:
    date = dt(y, 1, 1) + timedelta(d - 1)
    print(date)

    cad = a.CompositeAlbedoDay.run(date, 'scratch', 'scratch', 'scratch',
                                   'foo')
    cad.modis.plot()
