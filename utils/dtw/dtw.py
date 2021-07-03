from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

import os




# Dynamic Time Warping
class DTW():

    def __init__(self, file_path, word):
        self.word = word
        self.path = file_path
        self.createData(file_path)
        self.mfcc_var = self.MFCC(file_path)

    # Extract raw data from file path and store it into self.data (time domain)
    def createData(self, file_path):
        import wave
        import contextlib

        data = np.zeros((1, 32000))
        rate, wav_data = wavfile.read(file_path)
        with contextlib.closing(wave.open(file_path, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
            self.duration = duration
        self.data = wav_data

    # Routine for comparing two signals
    def compare(self, input_word_path):
        
        foreign_word_mfcc = self.MFCC(input_word_path)
        
        # Construct the DTW matrix
        self.dtwMatrix(foreign_word_mfcc)
        # Find the minimal path of reaching from top left end to bottom right end
        temp = np.zeros((self.mfcc_var.shape[0], foreign_word_mfcc.shape[0]), dtype=object)
        temp[:] = -1

        #cost = self.findLeastPath(temp, foreign_word_mfcc.shape[0])
        cost = self.findLeastPath_test(temp, foreign_word_mfcc.shape[0])
        
        return (cost[0][0][0]/cost[0][0][1])
        
    # Check
        
    # Construct the DTW matrix
    def dtwMatrix(self, foreign_word_mfcc):
        # foreign word in x axis (column)
        # training word in y axis (row)

        matrix = np.zeros((self.mfcc_var.shape[0], foreign_word_mfcc.shape[0]))

        row_index = 0
        column_index = 0
        
        for train_frame in self.mfcc_var:
            for foreign_frame in foreign_word_mfcc:
                matrix[row_index, column_index] = self.L2Distance(foreign_frame, train_frame)
                column_index += 1
                
            column_index = 0
            row_index += 1
            
        self.dtw_matrix = matrix

    def findLeastPath_test(self, memo_cost_matrix, foreign_len, row=0, column=0):
        train_len = self.mfcc_var.shape[0]

        if column == foreign_len - 1 and row == train_len - 1:
            memo_cost_matrix[row][column] = (self.dtw_matrix[row][column], 1)
            return memo_cost_matrix
        else:
            current_cost = self.dtw_matrix[row][column]

            if column == foreign_len - 1:
                if memo_cost_matrix[row + 1][column] == -1:
                    memo_cost_matrix = self.findLeastPath_test(memo_cost_matrix, foreign_len, row + 1, column)
                memo_cost_matrix[row][column] = (memo_cost_matrix[row + 1][column][0] + current_cost,
                                                 memo_cost_matrix[row + 1][column][1] + 1)
                return memo_cost_matrix
            elif row == train_len - 1:
                if memo_cost_matrix[row][column + 1] == -1:
                    memo_cost_matrix = self.findLeastPath_test(memo_cost_matrix, foreign_len, row, column + 1)
                memo_cost_matrix[row][column] = (memo_cost_matrix[row][column + 1][0] + current_cost,
                                                 memo_cost_matrix[row][column + 1][1] + 1)
                return memo_cost_matrix
            else:
                if memo_cost_matrix[row][column + 1] == -1:
                    temp_memo = self.findLeastPath_test(memo_cost_matrix, foreign_len, row, column + 1)
                    memo_cost_matrix[row][column + 1] = temp_memo[row][column + 1]
                if memo_cost_matrix[row + 1][column] == -1:
                    temp_memo = self.findLeastPath_test(memo_cost_matrix, foreign_len, row + 1, column)
                    memo_cost_matrix[row + 1][column] = temp_memo[row + 1][column]
                if memo_cost_matrix[row + 1][column + 1] == -1:
                    temp_memo = self.findLeastPath_test(memo_cost_matrix, foreign_len, row + 1, column + 1)
                    memo_cost_matrix[row + 1][column + 1] = temp_memo[row + 1][column + 1]
                
                right = memo_cost_matrix[row][column + 1][0]
                down = memo_cost_matrix[row + 1][column][0]
                diagonal = memo_cost_matrix[row + 1][column + 1][0]
                
                min_val = min(right, down, diagonal)
                
                if min_val == right:
                    step = memo_cost_matrix[row][column + 1][1]
                elif min_val == down:
                    step = memo_cost_matrix[row + 1][column][1]
                else:
                    step = memo_cost_matrix[row + 1][column + 1][1]
                
                memo_cost_matrix[row][column] = (min_val, step)

                return memo_cost_matrix
        
        
    def L2Distance(self, foreign, train):
        import math
        min_len = min(len(foreign), len(train))
        
        l2_ = 0
        
        for num in range(min_len):
            diff = foreign[num] - train[num]
            l2_ += (diff * diff)
        
        return math.sqrt(l2_)
            

    def MFCC(self, path):
        from utils.mfcc.feature import SpeechFeature
        import scipy.io.wavfile as wav
        (rate,sig) = wav.read(path)
        mfcc_feat = SpeechFeature(sig,rate)
        return mfcc_feat