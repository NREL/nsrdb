Contact:

Dr. Charles N. Long
NOAA ESRL GMD/CIRES
325 Broadway Street
Boulder, CO 80305

Chuck.long@noaa.gov

303-497-6056


***************************************************************

NOTICE: The Radiative Flux Analysis methodology is the result of 
many years of research by Dr. Charles N. Long. These data products 
are made available to you by NOAA ESRL GMD with the understanding that 
at the minimum you will clearly acknowledge the source, and where 
appropriate include Dr. Long as a co-author, as part of any presentation 
of results (including manuscripts for publication, talks, and posters).

****************************************************************

General description:

The Radiative Flux Analysis is a technique for using surface broadband 
radiation measurements for detecting periods of clear (i.e. cloudless)
skies, and using the detected clear-sky data to fit functions which are
then used to produce continuous clear-sky estimates. The clear-sky 
estimates and measurements are then used in various ways to infer cloud 
macrophysical properties. Detailed descriptions of the methodology are
given in the papers referenced, and a listing of the derived parameters, 
are given below.


===============================================================



Notes on the output files of the Radiative Flux Analysis code:

All data used as input are first tested with the QCRad methodology 
(Long and Shi, 2006, 2008). 

Various portions of the Radiative Flux Analysis methodology are described 
in  Long and Ackerman (2000), Long and Gaustad (2004), Long (2004, 2005), 
Long et al., (2006), Long and Turner (2008), Barnard and Long (2004), and 
Barnard et al. (2008). The clear-sky LW and LW effective sky cover 
techniques are based on the pioneering work of Marty and Philopona (2000) 
and Durr and Philipona, (2004), which in turn use a formulation from 
Brutsaert (1975). 

Some of this effort is a work-in-progress, and not all of the methodologies 
have undergone peer review. I am releasing these results to those interested 
with the understanding that some of these variables (as described below) are 
at this point preliminary results only.

 

Calculated variables that are considered "solid":

Estimates of clear-sky downwelling GlobalSW, DifSW, DirSW, upwelling SW; SW fractional 
sky cover; Cloud optical depth for sky cover > 0.95; effective cloud transmissivity; 
clear-sky downwelling LW, clear-sky upwelling SW, effective clear-sky emissivity.



Calculated variables that are considered "good":

LW sky cover 




Some calculated variables that are "work in progress":

Estimated clear-sky upwelling LW (if .flx. or .rfa files), cloudy sky radiating 
temperature (equivalent to NFOV IRT measurements), Cloud height estimates (CLOUD 
HEIGHT ESTIMATES VERY CRUDE, USE AT YOUR OWN RISK!!! SEE DETAILED DESCRIPTION BELOW.)



Notes:

The cloud optical depth estimates are based on a technique by Barnard et al. (2008). 
This technique, a derived relationship based on the results of Min and Harrison (1996) 
and Min et al. 2004 is officially only valid for overcast skies (sky cover > 0.90). Thus 
the current output includes cloud optical depth only for sky cover > 0.90 for now. 
Also, comparisons conducted as part of the ARM CLOWD project suggest that the 
Min and Harrison (1996) technique itself tends to overestimate the cloud optical depth 
for thinner clouds (Tau < 5) (Dave Turner, personal communication). Recent work using 
TWP-ICE data has prompted a change to using the total (global) SW in our formulation 
instead of the diffuse as in Min and Harrison, which appears to do well to compensate 
for this thin cloud overestimation (Barnard et al., 2008). Finally, an attempt is made 
to detect when the cloudiness present is likely to be ice clouds, for which an asymmetry 
parameter for ice (0.8 from Fu, 1996) should be used rather than the standard 0.87 used 
for liquid water clouds. The sky brightness temperature calculated from the downwelling 
LW using the Stephan-Boltzman relationship (Te) is compared to a limit temperature. The 
limit temperature is calculated using the effective clear-sky broadband LW emissivity (Ec) 
estimated by the Radiative Flux Analysis code (Long, 2004; Long and Turner, 2008) and the 
assumption that (1-Ec) is the extent to which clouds can influence the downwelling LW 
measurement. Then assuming a brightness temperature for the cloudy sky that contains a 
cloud at -40 C (where to first order only ice can exist), a limit is calculated as:

LWice = LWclr + Scv*(1-Ec)*sigma*Tice^4

Tlim = (LWice/sigma)^0.25 - 2.0

Where LWice is the limit in terms of LW irradiance, LWclr is the estimated clear-sky LW, 
Scv is the fractional sky cover, sigma is the Stephan-Boltzman constant, Tice is the 
cloudy-sky brightness temperature for a cloud at -40 C, and Tlim is the limit in terms 
of sky brightness temperature. Then for times with Te is less than Tlim, an asymmetry 
parameter of 0.8 is used in the calculation of cloud optical depth, else 0.87 is used. 
From analysis of ARM Darwin TWP-ICE data, Tice is set to 248 K to represent the ice 
cloudy sky brightness temperature.  


The estimated clear-sky downwelling LW is derived from a technique based on Brutsaert 
(1975). Unlike the Brutsaert formulation, we use the known clear-sky periods and the 
corresponding measured clear-sky downwelling LW to calculate lapse rate coefficients. 
These calculated lapse rate coefficients are then interpolated for cloudy periods, 
similar to the SW technique. Comparisons show that about 80% of the estimated clear-sky 
LW falls within 4 W/m^2 of the corresponding clear-sky measured LW, and within 8 W/m^2 
radiative transfer calculations (which themselves agree with clear-sky measutements at 
the 4 W/m^2 level) used as a comparison under cloudy skies (Long and Turner, 2008).  
There is a known "problem", however, in that the only information available for LW 
estimation is surface measurements. For those times of abrupt major changes in 
temperature or humidity profiles significantly differing from the data the lapse rate 
coefficients were determined from, such as cold front passages, the clear-sky LW estimates 
will exhibit greater error. This same problem occurs for model calculations due to the 
interpolation through time in between sonde profiles (Long and Turner, 2008). Fortunately, 
these conditions occur infrequently.


The LW effective sky cover is from a technique developed by Durr and Philipona (2004), 
but with some differences. Durr and Philipona use a climatologically derived and 
applied formulation for clear-sky effective broadband LW emissivity, whereas those 
here are derived from surrounding clear-sky data. In addition, Durr and Philipona 
use a calculation of downwelling LW standard deviation for the hour preceding the 
time of interest in their sky cover prediction, where here I use a running 21-minute 
standard deviation centered on the time of interest. The varible is deemed as the 
"effective LW sky cover" in that the downwelling LW at the surface is insensitive 
to high and thin clouds, thus the sky cover is essentially most representative of 
the amount of low and mid-level cloudiness (Long, 2004; Long and Turner, 2008). The 
original Durr and Philipona retrieval is in Oktas, so their inherent uncertainty is at 
least 1/8 of sky cover. I use a 7-minute running mean to smooth the results. ARM is 
working on fielding an Infrared Sky Imager that eventually should provide the data 
needed to refine the (or even develop a new) approach, similar to how I used TSI data 
to develop the SW sky cover technique. 


CSWup - There are identified problems associated with guesstimating upwelling SW 
measurements using only detected clear-sky measurements, and then interpolating fit 
coefficients as we do for the downwelling SW (Long, 2005). For instance, when it snows, 
it's cloudy, thus the "fit" is way off until the next "clear enough" day for fitting 
after the snow event. This introduces a large error during the period, and for times 
of snow melt. Data show that the bi-directional reflectance function also changes over 
time depending on the surface characteristics. Thus, the current procedure for estimating 
clear-sky upwelling SW is to look through the data and take a daily average for all data 
from 1100 through 1300 local standard time. This captures, at least on a daily basis, 
the major changes in surface albedo such as those from snow accumulation or snow melt. 
A second pass through the data then uses the "daily noon average" as a constant, and 
determines a function for any data that include at least 25% of the total SW produced 
by the direct component (i.e. significant direct sunlight producing the bi-directional 
nature of the albedo dependence) using the cosine of the solar zenith angle as the 
independent variable. Again, these fit coefficients are interpolated for days when 
insufficient direct SW data are available for fitting. The function is then multiplied  
times the estimated clear-sky SWdn to produce a continuous estimate of clear-sky SWup.  
My examination of these results so far suggest this technique does pretty much eliminate 
the "gotcha" of it always being cloudy when it snows, and does a better job than just 
multiplying the measured albedo (SWup/SWdn which often behaves erratically through time 
depending on whether the direct sun is blocked by cloud or not) times the clear-sky SWdn.
A paper on the technique is in progress. 


CLWup - In the "lw1" output files, when there are values other than "-9999.0" present 
they represent the actual measured upwelling LW when that time was determined to be 
effectively clear-sky for the broadband LW. For the "rfa and "flx" output files, 
the clear-sky upwelling LW uses the same detected SW and "LW effective" clear-
sky data to empirically derive fit coefficients that are again interpolated for cloudy 
periods (Long, 2005). In this case, since the upwelling LW is tied to the total surface 
energy exchange including latent and sensible heat, the independent variables used are the 
downwelling LW, the net SW, 2 meter relative humidity, and wind speed. These last are used 
as surrogates to help account for the unknown relative changes in surface sensible and 
latent heat partitioning with respect to the radiative terms. Comparisons show that over 
90% of the estimations agree with detected clear-sky LWup measurements to within 5 W/m^2. 
Though estimation of the accuracy of the interpolated values has yet to be investigated, 
visual inspection indicates that the results appear intuitively reasonable. The major 
assumption here is that the surface radiating temperature responds relatively quickly 
to changes in the radiative input to the surface, which is the case for land surfaces, 
but not so for water or snow surfaces. For water, such as oceans or swampy ground, 
the thermal mass of the water precludes rapid temperature response. For snow covered 
ground, a significant portion of the energy can be tied up in water phase change which 
then does not go into changing the surface skin temperature. Thus this technique does 
not work for water, snow, and ice surfaces. 


Cloud field temperature and height estimates - these are "work in progress". I use
the measured and clear-sky estimated LWdn, the LW sky cover amount, and Independent
Pixel Approximation arguments to estimate the LW effective radiating ("cloud") 
temperature. The uncertainty in this estimation is largely driven by the uncertainty 
associated with the LW effective sky cover. The value generated assumes a single layer 
of cloudiness covering the "LW sky cover" portion of the sky, and with uniform radiating 
properties. Thus this value is best described as an "effective cloud field radiating 
temperature" with all the assumptions that the word "effective" usually implies. 
Comparisons have shown that for LW sky cover of 50% or more, the retrieved radiating
temperatures show remarkable agreement with corresponding IRT measurements. However, the 
agreement rapidly degrades for LW sky < 50%, thus we limit these retrievals for times
when the LW sky cover is > 50%.

In addition, given a good cloud radiating temperature estimate, one 
must then figure out how to reasonably translate that temperature to a cloud height. 
I use here the difference between the estimated cloud field radiating temperature and 
the ambient air temperature, and a simple 10-degree-C-per-km lapse rate to estimate the 
effective cloud field radiating height. THIS IS VERY CRUDE!!! Note that the imaginary 
"radiating surface" relates approximately to about one optical depth into the cloud, 
and so is NOT located at the same height as the cloud physical boundary as would be 
determined by a lidar or cloud radar. Again, this is a work in progress, and to some 
degree these values are included in the output files as "place holders" for a time when 
better cloud height estimations might be possible through further development. USE 
THESE AT YOUR OWN RISK FOR NOW.   



====================================================================================
"YYYYMMDD.lw1" "YYYYMMDD.rfa" and "YYYYMMDD.flx" files:


Zdate		date in YYYYMMDD format, based on GMT
Ztim		time in hhmm format, based on GMT
Ldate		date in YYYYMMDD format, based on LST
Ltim		time in hhmm format, based on LST
CosZ		Cosine of the solar zenith angle
AU		earth-sun distance in AUs
SWdn		best estimate downwelling SW from sum or global pyranometer (W/m^2)
CSWdn		estimated clear-sky downwelling SW (W/m^2)
LWdn		downwelling LW from pyrgeometer (W/m^2)
CLWdn		estimated clear-sky downwelling LW (W/m^2)
SWup		upwelling SW from pyranometer (W/m^2)
CSWup		estimated clear-sky upwelling SW (W/m^2)
LWup		upwelling LW from pyrgeometer (W/m^2)
CLWup		estimated clear-sky upwelling LW (W/m^2)
DifSW		measured downwelling diffuse SW (W/m^2)
CDifSW		estimated clear-sky downwelling diffuse SW (W/m^2)
DirSW		measured downwelling direct SW (W/m^2)
CDirSW		estimated clear-sky downwelling direct SW (W/m^2)
ClrF		Clear sky flag, 1 if SW detected clear sky, 2 if LW detected, 9 if CLW>LW, 
		3 if only std and Ta-Te diff OK and ONLY LWup accepted as clear LWup [NOT LWdn!!!], else 0 if cloudy
TauF		Tau flag, 1 if liq g used, 2 if ice g used, 0 if not calculated
TlmF		T limit flag, 1 if SW Scv used, 2 if LW Scv used, 3 if avg Ec used, 4 if lim=0.965*Ta used, 
		5 if just config limit temp used, 0 if not calculated
LWScv		estimated effective LW fractional sky cover
SWScv		estimated fractional sky cover from SW
CldTau 		estimated effective visible cloud optical depth  (only for SWScv>0.95)  
CldTrn 		estimated effective SW cloud transmissivity (SWdn/CSWdn ratio)
TeLim		Ice cloud temp limit (K)
LWTe		Sky brightness temp from LWdn (K)   
CldTmp 		estimated effective cloud radiating temperature  
CldHgt 		estimated effective cloud radiating height    
Tair		air temperature (K)  
VPrs 		vapor pressure  (mb)      
RH		Relative Humidity (%)
RHfac  		RH adjustment to Ec        
Ec       	effective clear-sky LW emissivity
Wspd		Wind speed (same as input)

LWlw		(if included) Contribution to clear-sky LWup from LWdn term (W/m^2)
SWlw		(if included) Contribution to clear-sky LWup from SWnet term (W/m^2)
RHlw		(if included) Contribution to clear-sky LWup from RH term (W/m^2)
Wslw		(if included) Contribution to clear-sky LWup from Wspd term (W/m^2)



There may be other columns of data if the provider used the option to include 
up to 20 extra variables. Hopefully the column header abbreviations in this case 
are self-explanatory as to what the variables are...if not, contact me for more info.

NOTE: that no data quality testing have been applied 
to any of these extra variables by this processing. Hopefully data quality 
has been applied in producing the input files.



=====================================================================================
REFERENCES:

Barnard, J. C., and C. N. Long, (2004): A Simple Empirical Equation to Calculate Cloud Optical 
Thickness Using Shortwave Broadband Measurement, JAM, 43, 1057-1066.

Barnard, J. C., C. N. Long, E. I. Kassianov, S. A. McFarlane, J. M. Comstock, M. Freer, and G. 
M. McFarquhar (2008): Development and Evaluation of a Simple Algorithm to Find Cloud Optical 
Depth with Emphasis on Thin Ice Clouds, OASJ, 2, 46-55, doi: 10.2174/1874282300802010046.

Brutsaert, W., (1975): On a Derivable Formula for Longwave Radiation from Clear Skies, Water 
Resour. Res., 11(3), 742– 744.

Durr B. and R. Philipona, (2004): Automatic cloud amount detection by surface longwave downward
radiation measurements, JGR, 109, D05201, doi:10.1029/2003JD004182.
 
Fu, Q., (1996): An accurate parameterization of the solar radiative properties of cirrus clouds 
for climate models,  J. Climate, 9, 2058-2082.

Long, C. N. and T. P. Ackerman, (2000): Identification of Clear Skies from Broadband Pyranometer 
Measurements and Calculation of Downwelling Shortwave Cloud Effects, JGR, 105, No. D12, 15609-15626.

Long, C. N. and K. L. Gaustad, (2004): The Shortwave (SW) Clear-Sky Detection and Fitting 
Algorithm: Algorithm Operational Details and Explanations, Atmospheric Radiation Measurement Program 
Technical Report, ARM TR-004, Available via http://www.arm.gov.

Long, C. N., (2004): The Next Generation Flux Analysis: Adding Clear-sky LW and LW Cloud Effects, 
Cloud Optical Depths, and Improved Sky Cover Estimates, 14th ARM Science Team Meeting Proceedings, 
Albuquerque, New Mexico, March 22-26, 2004.

Long, C. N., (2005): On the Estimation of Clear-Sky Upwelling SW and LW, 15th ARM Science Team 
Meeting Proceedings, Daytona Beach, Florida, March 14-18, 2005.

Long, C. N., T. P. Ackerman, K. L. Gaustad, and J. N. S. Cole, (2006): Estimation of fractional sky 
cover from broadband shortwave radiometer measurements, JGR, 111, D11204, doi:10.1029/2005JD006475.

Long, C. N. and Y. Shi, (2006): The QCRad Value Added Product: Surface Radiation Measurement Quality 
Control Testing, Including Climatologically Configurable Limits, Atmospheric Radiation Measurement 
Program Technical Report, ARM TR-074, 69 pp., Available via http://www.arm.gov.

Long, C. N., and Y. Shi, (2008): An Automated Quality Assessment and Control Algorithm for Surface 
Radiation Measurements, TOASJ, 2, 23-37, doi: 10.2174/1874282300802010023. 

Long, C. N. and D. D. Turner (2008): A Method for Continuous Estimation of Clear-Sky Downwelling Longwave 
Radiative Flux Developed Using ARM Surface Measurements, J. Geophys. Res., 113, doi:10.1029/2008JD009936.

Marty, C. and R. Philipona, (2000): The Clear-sky Index to separate Clear-sky from Cloudy-sky 
Situations in Climate Research, GRL, 27, No. 19, 2649-2652.

Min, Q., Harrison, L. C., (1996): Cloud Properties Derived from Surface MFRSR Measurements and 
Comparison with GOES Results at the ARM SGP Site, Geophysical Research Letters, Vol. 23, 
pp. 1641-1644.
