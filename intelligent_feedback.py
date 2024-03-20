'''Import necessary libraries'''
'''Make the necessary imports'''
import os
import numpy as np
from numpy import mean
import random
from math import ceil, floor
import json
from scipy.signal import hamming 
import pandas as pd
import numpy as np
import sys
import parselmouth
from scipy.io import wavfile
from scipy.signal import medfilt
import math
from pydub import AudioSegment
from parselmouth.praat import call
from pydub import AudioSegment

# Compute the short-term energy and spectral centroid of the signal
def ShortTimeEnergy(signal, windowLength,step):
    signal = signal / max(signal)
    curPos = 1;
    L = len(signal)
    
    from math import floor, ceil
    numOfFrames = floor((L-windowLength)/step) + 1;
    
    E = np.zeros((numOfFrames,1))
    
    for i in range(numOfFrames):
        window = signal[ceil(curPos):ceil(curPos+ windowLength-1)];
        E[i] = (1/(windowLength)) * sum(abs(np.power(window,2)));
        curPos = curPos + step;
        
    return E

# Compute the short-term energy and spectral centroid of the signal
def SpectralCentroid(signal, windowLength, step, fs):
    windowLength = ceil(windowLength);
    
    signal = signal / max(abs(signal));
    
    curPos = 0;
    L = len(signal);
    numOfFrames = floor((L-windowLength)/step) + 1;
    
    H = hamming(windowLength);
    m = np.array([(fs/(2*windowLength))*float(i+1) for i in range(windowLength)])
    m = m.transpose()
    C = np.zeros((numOfFrames,1))
    
    for i in range(numOfFrames):

        # Broadcast length check
        broadcast_length = len(signal[ceil(curPos):ceil(curPos+windowLength)])

        window = H[0:broadcast_length]*signal[ceil(curPos):ceil(curPos+windowLength)]
        FFT = (abs(np.fft.fft(window,2*windowLength)));
        FFT = FFT[0:windowLength];  
        FFT = FFT / max(FFT);
        C[i] = sum(m*FFT)/sum(FFT);
        if (sum(np.power(window,2))<0.010):
            C[i] = 0.0

        curPos = curPos + step;
        
    C = C / (fs/2);
    
    return C

def findMaxima(f, step):
    
    countMaxima = 0

    Maxima = [[], []]

    for i in range(0, len(f)-step-1):  # for each element of the sequence:
        if (i>step):
            if (( mean(f[i-step:i])< f[i]) and ( mean(f[i+1:i+step+1])< f[i])):  
                # IF the current element is larger than its neighbors (2*step window)
                # --> keep maximum:
                countMaxima = countMaxima + 1;
                Maxima[0].append(i);
                Maxima[1].append(f[i]);
        else:
            if (( mean(f[0:i+1])<= f[i]) and ( mean(f[i+1:i+step])< f[i])):
                # IF the current element is larger than its neighbors (2*step window)~
                # --> keep maximum:
                countMaxima = countMaxima + 1;
                Maxima[0].append(i);
                Maxima[1].append(f[i]);
                

    #
    # STEP 2: post process maxima:
    #

    MaximaNew = [[], []];
    countNewMaxima = 0;
    i = 0; # Python indexing starts from 0
    while (i<countMaxima-1):
        # get current maximum:

        curMaxima = Maxima[0][i];
        curMavVal = Maxima[1][i];

        tempMax = [Maxima[0][i]];
        tempVals = [Maxima[1][i]];

        # search for "neighbourh maxima":
        while ((i<countMaxima-1) and ( Maxima[0][i+1] - tempMax[-1] < step / 2)):
            tempMax.append(Maxima[0][i]);
            tempVals.append(Maxima[1][i]);
            i = i + 1;

        # find the maximum value and index from the tempVals array:
        # MI = findCentroid(tempMax, tempVals); MM = tempVals(MI);

        MM = max(tempVals);
        MI = np.argmax(tempVals);

        if (MM>0.02*mean(f)): # if the current maximum is "large" enough:
            # keep the maximum of all maxima in the region:
            MaximaNew[0].append(tempMax[MI]) 
            MaximaNew[1].append(f[MaximaNew[0][countNewMaxima]]);

            countNewMaxima = countNewMaxima + 1;   # add maxima

        tempMax = [];
        tempVals = [];

        ## Update the counter
        i = i + 1;

    Maxima = MaximaNew;
    countMaxima = countNewMaxima;   
    
    return (Maxima, countMaxima)


'''Main function call'''
def pause_main(file_location):

    '''Read the file'''
    fs, x = wavfile.read(file_location)

    N = len(x);
    t = [i/fs for i in range(N)]; # Transform into time domain


    # Window length and step (in seconds)
    win = 0.050
    step = 0.050

    ## Threshold Estimation
    Weight = 5 # Used in the threshold estimation method

    Eor = ShortTimeEnergy(x, win*fs, step*fs)
    # Compute spectral centroid
    Cor = SpectralCentroid(x, win*fs, step*fs, fs)

    # Apply median filtering in the feature sequences (twice), using 5 windows:
    # (i.e., 250 mseconds)
    E = medfilt(medfilt([i[0] for i in Eor.tolist()], 5), 5)

    C = medfilt(medfilt([i[0] for i in Cor.tolist()], 5), 5)

    # Get the average values of the smoothed feature sequences:
    E_mean = np.mean(E);
    Z_mean = np.mean(C);

    # Find energy threshold
    [HistE, X_E] = np.histogram(E, bins = round(len(E) / 10));  # histogram computation
    X_E = np.array([(X_E[idx]+X_E[idx+1])/2 for idx in range(len(X_E)-1)])

    [MaximaE, countMaximaE] = findMaxima(HistE, 3);

    if (len(MaximaE[0])>=2): # if at least two local maxima have been found in the histogram:
        T_E = (Weight*X_E[MaximaE[0][0]]+X_E[MaximaE[0][1]]) / (Weight+1); # ... then compute the threshold as the weighted average between the two first histogram's local maxima.
    else:
        T_E = E_mean / 2;

    # Find spectral centroid threshold:
    [HistC, X_C] = np.histogram(C, round(len(C) / 10));
    X_C = np.array([(X_C[idx]+X_C[idx+1])/2 for idx in range(len(X_C)-1)])

    [MaximaC, countMaximaC] = findMaxima(HistC, 3);
    if (len(MaximaC[0])>=2):
        T_C = (Weight*X_C[MaximaC[0][0]]+X_C[MaximaC[0][1]]) / (Weight+1);
    else:
        T_C = Z_mean / 2;

    Flags1 = (E>=T_E);
    Flags2 = (C>=T_C);
    
    '''Check if array broadcasting doesn't have issues.'''
    if Flags1.shape[0]!=Flags2.shape[0]:
        if Flags1.shape[0]>Flags2.shape[0]:
            Flags1 = Flags1[0:Flags2.shape[0]]

        if Flags2.shape[0]>Flags1.shape[0]:
            Flags2 = Flags1[0:Flags1.shape[0]]
        
    flags = Flags1 & Flags2

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # %  SPEECH SEGMENTS DETECTION
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    count = 0;
    WIN = 5;
    Limits = [[], []];
    while (count < len(flags)): # while there are windows to be processed:
        # initilize:
        curX = [];
        countTemp = 1;
        # while flags=1:
        while ((flags[count]==True) and (count < len(flags))):
            if (countTemp==1): # if this is the first of the current speech segment:
                Limit1 = round((count+1-WIN)*step*fs)+1; # set start limit:
                if (Limit1<1):
                    Limit1 = 1;         

            count = count + 1; # increase overall counter
            
            if count==len(flags):
                break
            
            countTemp = countTemp + 1; # increase counter of the CURRENT speech segment

        if (countTemp>1): # if at least one segment has been found in the current loop:
            Limit2 = round((count+1+WIN)*step*fs); # set end counter
            if (Limit2>len(x)):
                Limit2 = len(x);

            Limits[0].append(Limit1-1);
            Limits[1].append(Limit2-1);

        count = count + 1; # increase overall counter

    # %%%%%%%%%%%%%%%%%%%%%%%
    # % POST - PROCESS      %
    # %%%%%%%%%%%%%%%%%%%%%%%
    # % A. MERGE OVERLAPPING SEGMENTS:
    RUN = 1;
    while RUN==1:
        RUN = 0;
        for i in range(0, len(Limits[0])-1): # for each segment
            if (Limits[1][i]>=Limits[0][i+1]):
                RUN = 1;
                Limits[1][i] = Limits[1][i+1];
                Limits[0].pop(i+1)
                Limits[1].pop(i+1)
                break

    # B. Get final segments:
    segments = [];
    for i in range(0, len(Limits[0])):
        segments.append(x[Limits[0][i]:Limits[1][i]]); 

    # Record pause positions
    pause_positions = [[0 for i in range(len(Limits[0]))], [0 for i in range(len(Limits[0]))]]

    for i in range(0, len(segments)): # Number of segments = Number of Limits
        if i==0:
            if Limits[0][0]!=0:
                pause_positions[0][0] = 0;
                pause_positions[1][0] = Limits[0][0];
            else:
                pause_positions[0][0] = Limits[0][0];
                pause_positions[1][0] = Limits[1][0];

        else:
            pause_positions[0][i] = Limits[1][i-1];
            pause_positions[1][i] = Limits[0][i];

    # Print pause positions and durations
    pause_info = []
    for i in range(0, len(pause_positions[0])):        
        pause_info.append({
                'pause_start': t[pause_positions[0][i]],
                'pause_finish': t[pause_positions[1][i]],
                'pause_duration': t[pause_positions[1][i]] - t[pause_positions[0][i]]
            })
    
    return pause_info, t

def pause_feedback(pause_score):
    
    text = ''
    text_list = []
    if pause_score>=90:
        temp_lst = ['Great job!', 'Great work!', 'Excellent!', 'Bravo!', 'Amazing work!', 'Exemplary work!', 
                   ]
        temp_lst2 = ['You are doing well in reducing awkward pauses.', 'The less awkward pauses you have, the better your speech.',
                    'With fewer awkward pauses the flow of your speech is not broken and its easy to understand.',
                    'You really prepared your speech well as visible by your lack of awkward pauses.']
        random.shuffle(temp_lst)
        random.shuffle(temp_lst2)
        text += """{} {} But remember pauses are not all bad. Great speakers use pause for introducing dramatic effect and introduce smooth transitions between their speech. Just imagine an anchor on stage saying, "Please, welcome the one and only.... Amitabh Bachhan." You will notice that there is often a pause before introducing Amitabh Bachhan in order to create suspense. Now you too practice introducing pauses purposefully to increase the quality of your speech.""".format(temp_lst[0], temp_lst2[0])
        text_list += [temp_lst[0], temp_lst2[0], """But remember pauses are not all bad. Great speakers use pause for introducing dramatic effect and introduce smooth transitions between their speech. Just imagine an anchor on stage saying, "Please, welcome the one and only.... Amitabh Bachhan." You will notice that there is often a pause before introducing Amitabh Bachhan in order to create suspense. Now you too practice introducing pauses purposefully to increase the quality of your speech."""]
    
    elif pause_score>=80:
        temp_lst = ['Good job!', 'Good work!', 'Well done!', 'Nice!']
        temp_lst2 = ['You are doing well but can improve in reducing awkward pauses.', 
                     'Your speech flow is good because you lack large number of awkward pauses. Nonetheless, you can do better.',
                    'You are on path of success, work even more to reduce the number of clumsy pauses in your speech.',
                    'Work on your speech some more until you minimize the number of unexpected pauses you take.']
        random.shuffle(temp_lst)
        random.shuffle(temp_lst2)
        text += """{} {} When you have unexpected pauses then it can distract your audience. A pause can be powerful if it is planned well in speech such as an introduction (Here comes Mr. Akshay), or an exclamation (Wow!). However, unplanned pauses breaks your sentences in between and confuses your audience.""".format(temp_lst[0], temp_lst2[0])
        text_list += [temp_lst[0], temp_lst2[0], """When you have unexpected pauses then it can distract your audience. A pause can be powerful if it is planned well in speech such as an introduction (Here comes Mr. Akshay), or an exclamation (Wow!). However, unplanned pauses breaks your sentences in between and confuses your audience."""]
        
    elif pause_score<80:
        temp_lst = ['Satisfactory!', 'OK!', 'Not great!', 'You can do better!']
        temp_lst2 = ['You need to improve in reducing clumsy pauses.', 
                     'You have unusually large number of unexpected pauses in your speech.',
                    'You should work more to reduce the number of long and short pauses from your speech.',
                    'Your quality of speech can be improved by reducing the number of these awkward pauses.']
        random.shuffle(temp_lst)
        random.shuffle(temp_lst2)
        text += """{} {} A large number of pauses in a speech is a sign of either lack of confidence or lack of preparation. Sometimes both. Try to practice your speech in advance in front of mirror or members of your family without any notes to improve confidence. Having less unwanted pauses will make you appear more confident in front of your audience and will also make you easy to understand.""".format(temp_lst[0], temp_lst2[0])
        text_list += [temp_lst[0], temp_lst2[0], """A large number of pauses in a speech is a sign of either lack of confidence or lack of preparation. Sometimes both. Try to practice your speech in advance in front of mirror or members of your family without any notes to improve confidence. Having less unwanted pauses will make you appear more confident in front of your audience and will also make you easy to understand."""]
        
    return text, text_list


def pace_feedback(pace_score, articulation_rate):
    
    text = ''
    text_list = []
    if pace_score>=85:
        temp_lst = ['Great job!', 'Great work!', 'Excellent!', 'Bravo!', 'Amazing work!', 'Exemplary work!', 
                   ]
        
        random.shuffle(temp_lst)
        text += """{} Your pace is in recommended limit and will be easy to follow for most of your audience.""".format(temp_lst[0])
        text_list += [temp_lst[0], """Your pace is in recommended limit and will be easy to follow for most of your audience."""]
    
    elif pace_score>=60:
        temp_lst = ['Good job!', 'Good work!', 'Well done!', 'Nice!']
        random.shuffle(temp_lst)
        text += """{} Your pace needs more work to make it more interesting for the audience. Slow speaking overall bores the audience and their attention wanders away.""".format(temp_lst[0])
        text_list += [temp_lst[0], """Your pace needs more work to make it more interesting for the audience. Slow speaking overall bores the audience and their attention wanders away."""]
        
    elif pace_score<60:
        temp_lst = ['Satisfactory!', 'OK!', 'Not great!', 'You can do better!']
        random.shuffle(temp_lst)
        
        if articulation_rate>240:
            text += """{} Your pace is too fast for the audience to follow. Try practicing tongue twisters to slow yourself down which will make your enunciation better and clearer.""".format(temp_lst[0])
            text_list += [temp_lst[0], """Your pace is too fast for the audience to follow. Try practicing tongue twisters to slow yourself down which will make your enunciation better and clearer."""]
        else:
            text += """{} Your pace is too slow and dull which will lose audience attention. Speed up the pace by trying speed reading with speaking which will develop the muscular strength and skill to speak faster.""".format(temp_lst[0])
            text_list += [temp_lst[0], """Your pace is too slow and dull which will lose audience attention. Speed up the pace by trying speed reading with speaking which will develop the muscular strength and skill to speak faster."""]
        
    return text, text_list


def speech_rate(filename):
    cols = ['soundname', 'nsyll', 'npause', 'dur(s)', 'phonationtime(s)', 'speechrate(nsyll / dur)', 'articulation '
        'rate(nsyll / phonationtime)', 'ASD(speakingtime / nsyll)']
    
    silencedb = -25
    mindip = 2
    minpause = 0.3
    sound = parselmouth.Sound(filename)
    originaldur = sound.get_total_duration()
    intensity = sound.to_intensity(50)
    start = call(intensity, "Get time from frame number", 1)
    nframes = call(intensity, "Get number of frames")
    end = call(intensity, "Get time from frame number", nframes)
    min_intensity = call(intensity, "Get minimum", 0, 0, "Parabolic")
    max_intensity = call(intensity, "Get maximum", 0, 0, "Parabolic")

    # get .99 quantile to get maximum (without influence of non-speech sound bursts)
    max_99_intensity = call(intensity, "Get quantile", 0, 0, 0.99)

    # estimate Intensity threshold
    threshold = max_99_intensity + silencedb
    threshold2 = max_intensity - max_99_intensity
    threshold3 = silencedb - threshold2
    if threshold < min_intensity:
        threshold = min_intensity

    # get pauses (silences) and speakingtime
    textgrid = call(intensity, "To TextGrid (silences)", threshold3, minpause, 0.1, "silent", "sounding")
    silencetier = call(textgrid, "Extract tier", 1)
    silencetable = call(silencetier, "Down to TableOfReal", "sounding")
    npauses = call(silencetable, "Get number of rows")
    speakingtot = 0
    for ipause in range(npauses):
        pause = ipause + 1
        beginsound = call(silencetable, "Get value", pause, 1)
        endsound = call(silencetable, "Get value", pause, 2)
        speakingdur = endsound - beginsound
        speakingtot += speakingdur

    intensity_matrix = call(intensity, "Down to Matrix")
    # sndintid = sound_from_intensity_matrix
    sound_from_intensity_matrix = call(intensity_matrix, "To Sound (slice)", 1)
    # use total duration, not end time, to find out duration of intdur (intensity_duration)
    # in order to allow nonzero starting times.
    intensity_duration = call(sound_from_intensity_matrix, "Get total duration")
    intensity_max = call(sound_from_intensity_matrix, "Get maximum", 0, 0, "Parabolic")
    point_process = call(sound_from_intensity_matrix, "To PointProcess (extrema)", "Left", "yes", "no", "Sinc70")
    # estimate peak positions (all peaks)
    numpeaks = call(point_process, "Get number of points")
    t = [call(point_process, "Get time from index", i + 1) for i in range(numpeaks)]

    # fill array with intensity values
    timepeaks = []
    peakcount = 0
    intensities = []
    for i in range(numpeaks):
        value = call(sound_from_intensity_matrix, "Get value at time", t[i], "Cubic")
        if value > threshold:
            peakcount += 1
            intensities.append(value)
            timepeaks.append(t[i])

    # fill array with valid peaks: only intensity values if preceding
    # dip in intensity is greater than mindip
    validpeakcount = 0
    currenttime = timepeaks[0]
    currentint = intensities[0]
    validtime = []

    for p in range(peakcount - 1):
        following = p + 1
        followingtime = timepeaks[p + 1]
        dip = call(intensity, "Get minimum", currenttime, timepeaks[p + 1], "None")
        diffint = abs(currentint - dip)
        if diffint > mindip:
            validpeakcount += 1
            validtime.append(timepeaks[p])
        currenttime = timepeaks[following]
        currentint = call(intensity, "Get value at time", timepeaks[following], "Cubic")

    # Look for only voiced parts
    pitch = sound.to_pitch_ac(0.02, 30, 4, False, 0.03, 0.25, 0.01, 0.35, 0.25, 450)
    voicedcount = 0
    voicedpeak = []

    for time in range(validpeakcount):
        querytime = validtime[time]
        whichinterval = call(textgrid, "Get interval at time", 1, querytime)
        whichlabel = call(textgrid, "Get label of interval", 1, whichinterval)
        value = pitch.get_value_at_time(querytime) 
        if not math.isnan(value):
            if whichlabel == "sounding":
                voicedcount += 1
                voicedpeak.append(validtime[time])

    # calculate time correction due to shift in time for Sound object versus
    # intensity object
    timecorrection = originaldur / intensity_duration

    # Insert voiced peaks in TextGrid
    call(textgrid, "Insert point tier", 1, "syllables")
    for i in range(len(voicedpeak)):
        position = (voicedpeak[i] * timecorrection)
        call(textgrid, "Insert point", 1, position, "")

    # return results
    speakingrate = voicedcount / originaldur
    articulationrate = voicedcount / speakingtot
    npause = npauses - 1
    asd = speakingtot / voicedcount
    speechrate_dictionary = {'soundname':filename,
                             'nsyll':voicedcount,
                             'npause': npause,
                             'dur(s)':originaldur,
                             'phonationtime(s)':intensity_duration,
                             'speechrate(nsyll / dur)': speakingrate,
                             "articulation rate(nsyll / phonationtime)":articulationrate,
                             "ASD(speakingtime / nsyll)":asd}
    return speechrate_dictionary


def compute_articulation_rate(file_location):
    try:
        speechrate_dictionary = speech_rate(file_location)
        articulation_rate = speechrate_dictionary['articulation rate(nsyll / phonationtime)']
    except:
        articulation_rate = 0
        
    return articulation_rate

def return_articulation_rate(file_location):
    
    articulation_rate = compute_articulation_rate(file_location)
    articulation_rate = articulation_rate*60
    
    return articulation_rate

def energy_feedback(energy_score):
    
    text = ''
    text_list = []
    if energy_score>=90:
        temp_lst = ['Great job!', 'Great work!', 'Excellent!', 'Bravo!', 'Amazing work!', 'Exemplary work!', 
                   ]
        temp_lst2 = ["Your energy is commendable.', 'Your voice carries energy and momentum that can catch people's attention .",
                    'Your power reflects the confidence you have in yourself and your preparation',
                    'Your energy is high and that is a good note to start a speech on.']
        random.shuffle(temp_lst)
        random.shuffle(temp_lst2)
        text += """{} {}. While a good energy is nice to have, being too loud will also turn off your audience. It is best to start with strong energy and then bring variation along with vocal variety. It would be best to start with high energy, then maintain a consistent, normal volume and soften up if the speech demands it such as when showing despair and grief.""".format(temp_lst[0], temp_lst2[0])
        text_list += [temp_lst[0], temp_lst2[0], """While a good energy is nice to have, being too loud will also turn off your audience. It is best to start with strong energy and then bring variation along with vocal variety. It would be best to start with high energy, then maintain a consistent, normal volume and soften up if the speech demands it such as when showing despair and grief."""]
    
    elif energy_score>=60:
        temp_lst = ['Good job!', 'Good work!', 'Well done!', 'Nice!']
        temp_lst2 = ['You are doing well but can still improve your overall energy.', 
                     'Your speaking voice is commanding but there is room for further improvement.',
                    "You can attract people's attention but with further practice you will be able to hold it for longer duration.",
                    "Your speaking voice has potential but with further practice it can be perfected for even live auditorium."]
        random.shuffle(temp_lst)
        random.shuffle(temp_lst2)
        text += """{} {}. Having a good power is a great way to start a speech as it catches audience attention and create a positive, energetic vibe. You can improve your energy by practicing taking long breaths and speaking out sounds such as Om, Do, Re, Me, Fa, So, La, Ti for extended periods of time.""".format(temp_lst[0], temp_lst2[0])
        text_list += [temp_lst[0], temp_lst2[0], """Having a good power is a great way to start a speech as it catches audience attention and create a positive, energetic vibe. You can improve your energy by practicing taking long breaths and speaking out sounds such as Om, Do, Re, Me, Fa, So, La, Ti for extended periods of time."""]
        
    elif energy_score<60:
        temp_lst = ['Satisfactory!', 'OK!', 'Not great!', 'You can do better!']
        temp_lst2 = ['Your needs further improvement as you would not be audible to many people.', 
                     "Your speech lacks the gravity to catch people's attention.",
                    'You should work more on to increasing the strength of throat muscle to improve your vocal strength.',
                    'Your voice is too soft and uninspiring at the moment.']
        random.shuffle(temp_lst)
        random.shuffle(temp_lst2)
        text += """{} {}. Speaking with energy is vital for audience engagement. A lot of power comes from your lungs and breathing. Try practicing breathing and speaking from your stomach which means always take deep breath to fill all your diaphragm. Beyond that speak out loud sounds such as Om, Do, Re, Me, Fa, So, La, Ti for extended periods of time.""".format(temp_lst[0], temp_lst2[0])
        text_list += [temp_lst[0], temp_lst2[0], """Speaking with energy is vital for audience engagement. A lot of power comes from your lungs and breathing. Try practicing breathing and speaking from your stomach which means always take deep breath to fill all your diaphragm. Beyond that speak out loud sounds such as Om, Do, Re, Me, Fa, So, La, Ti for extended periods of time."""]
        
    return text, text_list

def return_energy_score(file_location):

    snd = parselmouth.Sound(file_location)
    intensity = snd.to_intensity()
    item = intensity.values.T
    
    new_item = []
    for val in item:
        if (val>=50) and (val<=75):
            new_item.append(1)
        elif (val<50) and (val>=30):
            new_item.append(-1)
        elif val>75:
            new_item.append(2)
        else:
            new_item.append(0)
    
    energy_score = sum([i for i in new_item if i!=0])/(len([i for i in new_item if i!=0]))*100
    
    return energy_score

def return_pitch_score(file_location):

    snd = parselmouth.Sound(file_location)
    pitch = snd.to_pitch()
    pitch = pitch.selected_array['frequency']
    
    new_item = []
    for val in pitch:
        if (val>=200) and (val<=300):
            new_item.append(1)
        elif val<200:
            new_item.append(0)
        else:
            new_item.append(2)
            
    pitch_score =  sum(new_item)/(len(new_item))*100
    
    return pitch_score

def pitch_feedback(pitch_score):
    
    text = ''
    text_list = []
    if pitch_score>=90:
        temp_lst = ['Great job!', 'Great work!', 'Excellent!', 'Bravo!', 'Amazing work!', 'Exemplary work!', 
                   ]
        temp_lst2 = ['Your use of vocal variety is commendable.', 'Your intonation and emphasis is adding beauty to your speech.',
                    'Your vocal variety will make listener want to listen to you.',
                    'Your pitch variation helps in expanding the meaning behind your words.']
        random.shuffle(temp_lst)
        random.shuffle(temp_lst2)
        text += """{} {} Afterall, a good vocal variety helps in communicating clear message to your audience about your intention. This makes you more memorable as a speaker and makes it easy to follow the key message you are trying to emphasize. However, beware of too much vocal variation as that will appear phony and pretentious.""".format(temp_lst[0], temp_lst2[0])
        text_list += [temp_lst[0], temp_lst2[0], """Afterall, a good vocal variety helps in communicating clear message to your audience about your intention. This makes you more memorable as a speaker and makes it easy to follow the key message you are trying to emphasize. However, beware of too much vocal variation as that will appear phony and pretentious."""]
    
    elif pitch_score>=60:
        temp_lst = ['Good job!', 'Good work!', 'Well done!', 'Nice!']
        temp_lst2 = ['You are doing well but can improve your vocal variation.', 
                     'Your speech intonation is good because you are able to modify the pitch of your voice. Nonetheless, you can do better.',
                    'Your vocal variation deserves praise, but work on bringing more emotions into your speech to make it better.',
                    'Your speech can illustrate your emotions and thus meaning behind your words. However, record yourself and listen to improve your vocal variation even more.']
        random.shuffle(temp_lst)
        random.shuffle(temp_lst2)
        text += """{} {}. The level of vocal variety you can bring to your speech adds more life into your speech. It not only helps to illustrate your meaning better but also improves your overall tone which makes it pleasant for audience to listen to you.""".format(temp_lst[0], temp_lst2[0])
        text_list += [temp_lst[0], temp_lst2[0], """The level of vocal variety you can bring to your speech adds more life into your speech. It not only helps to illustrate your meaning better but also improves your overall tone which makes it pleasant for audience to listen to you."""]
        
    elif pitch_score<60:
        temp_lst = ['There is much room for improvement!', 'Lot of progress needed', 'Not great!', 'You can do better!']
        temp_lst2 = ['You need to improve your vocal variation.', 
                     'Your speech yet lacks the variation and appeal to excite the audience to listen to you.',
                    'You should work more on to infusing emotions into your words to make it more exciting and less monotonous.',
                    'Your speaking style yet is quite montonous. Try practicing elongating your vowels and also try to infuse more emotions into your words.']
        random.shuffle(temp_lst)
        random.shuffle(temp_lst2)
        text += """{} {}. Having poor vocal range signifies lack of emotion and excitement in a speech. This can bore audience and lose them to stop listening. A good way to improve vocal variety is to practice your favorite movie dialogues especially which are emotional like patriotic movies. This will start to build your natural tendency to bring different flavors in your speech.""".format(temp_lst[0], temp_lst2[0])
        text_list += [temp_lst[0], temp_lst2[0], """Having poor vocal range signifies lack of emotion and excitement in a speech. This can bore audience and lose them to stop listening. A good way to improve vocal variety is to practice your favorite movie dialogues especially which are emotional like patriotic movies. This will start to build your natural tendency to bring different flavors in your speech."""]
        
    return text, text_list

def check_blank_audio(file_location):
    '''Read the file and check if it is blank'''
    fs, x = wavfile.read(file_location.replace(".aac", ".wav"))
        
    if max(x)==0:
        return 'Blank'
    
    else:
        return "Not Blank"


def main(file_location):
    
    try:
    
        if '.wav' in file_location:
            file_location = file_location
        else:
            src = file_location
            dest = src.replace(".aac", ".wav")
    #         sound = AudioSegment.from_file(src, format = 'aac')

            ##-------------------------
            import subprocess # Using bash utility for conversion of aac to wav and then reopning it in pydub to check it's mono

            try:
                os.remove(dest) # If same wave file exists in the destination
            except:
                pass 

            command = "ffmpeg -i " + src + " " + dest
            process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()

            sound = AudioSegment.from_file(dest, format = 'wav')
            if sound.channels>1:
                sound = sound.set_channels(1)

            sound.export(dest, format="wav")

            file_location = dest

            #-------------------------------


        output_dict = {} # Initialize the output variable

        ## Check blank
        check_blank = check_blank_audio(file_location)
        if check_blank=='Blank':
            output_dict['Pause'] = {'Score': 0, 'Text': ['Blank audio']}
            output_dict['Pace'] = {'Score': 0, 'Text': ['Blank audio']}
            output_dict['Power'] = {'Score': 0, 'Text': ['Blank audio']}
            output_dict['Pitch'] = {'Score': 0, 'Text': ['Blank audio']}

            output_dict['Blank'] = 'YES'
            print(json.dumps(output_dict))
            return 0 # Stop the processing of the code

        else:
            output_dict['Blank'] = 'NO'

        ## Pause Feedback
        pause_info, time_arr = pause_main(file_location)

        short_pauses = [(i['pause_start'], i['pause_finish'], i['pause_duration']) for i in pause_info if (i['pause_duration']>=0.5 and i['pause_duration']<1)]
        long_pauses = [(i['pause_start'], i['pause_finish'], i['pause_duration']) for i in pause_info if (i['pause_duration']>=1)]

        pause_score = int(np.round(((1 - sum([i['pause_duration'] for i in pause_info])/time_arr[-1]))*100))

        text, text_list = pause_feedback(pause_score)

        if len(short_pauses)<=1 and len(long_pauses)<=1:
            text += ' Overall, you took {} short pause and {} long pause.'.format(len(short_pauses), len(long_pauses))
        elif len(short_pauses)<=1:
            text += ' Overall, you took {} short pause and {} long pauses.'.format(len(short_pauses), len(long_pauses))
        elif len(long_pauses)<=1:
            text += ' Overall, you took {} short pauses and {} long pause.'.format(len(short_pauses), len(long_pauses))
        else:
            text += ' Overall, you took {} short pauses and {} long pauses.'.format(len(short_pauses), len(long_pauses))


        output_dict['Pause'] = {
            'Score': pause_score,
            'Text': text_list
        }

         ## Pace Feedback
        articulation_rate = return_articulation_rate(file_location)

        pace_score = np.round((1 - (abs(240-articulation_rate))/240)*100)

        text, text_list = pace_feedback(pace_score, articulation_rate)

        output_dict['Pace'] = {
            'Score': pace_score,
            'Text': text_list
        }

        ## Power Feedback
        energy_score = int(np.round(return_energy_score(file_location)))

        if energy_score>100: 
            energy_score = random.randint(90, 95)

        if energy_score<0:
            energy_score = random.randint(10, 20)

        text, text_list = energy_feedback(energy_score)

        output_dict['Power'] = {
            'Score': energy_score,
            'Text': text_list
        }

        ## Pitch Feedback
        pitch_score = int(np.round(return_pitch_score(file_location)))
        
        if pitch_score>100: 
            pitch_score = random.randint(90, 95)

        if pitch_score<20: # Hack earlier it was <0 and randint(0,10)
            pitch_score = random.randint(20, 30)
            
            
        text, text_list = pitch_feedback(pitch_score)



        output_dict['Pitch'] = {
            'Score': pitch_score,
            'Text': text_list
        }

        # print(json.dumps(output_dict))
        return output_dict
        
    except:
        # print(json.dumps({'Error': "Unexpected error has occurred!"}))
        return {'Error': "Unexpected error has occurred!"}

# file_location = sys.argv[1]

# main(file_location)