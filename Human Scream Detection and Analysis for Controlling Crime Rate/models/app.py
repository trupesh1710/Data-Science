# import tkinter as tk
# from tkinter import filedialog, messagebox
# import librosa
# import numpy as np
# import joblib
# import pandas as pd

# # --- Load the saved model and scaler ---
# try:
#     model = joblib.load('logistic_regression_model.joblib')
#     scaler = joblib.load('scaler_logistic.joblib')
# except FileNotFoundError:
#     # If files are not found, show an error on startup
#     messagebox.showerror("Error", "Model or scaler file not found. Please run the saving script first.")
#     model = None
#     scaler = None

# # --- Function for audio processing and feature extraction (Updated) ---
# def extract_features(file_path):
#     TARGET_SR = 22050
#     AUDIO_DURATION = 5
#     NUM_MFCC = 20
#     SAMPLES_PER_TRACK = TARGET_SR * AUDIO_DURATION
    
#     try:
#         signal, sr = librosa.load(file_path, sr=TARGET_SR, duration=AUDIO_DURATION)
#         if len(signal) < SAMPLES_PER_TRACK:
#             signal = np.pad(signal, (0, SAMPLES_PER_TRACK - len(signal)), 'constant')
        
#         mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=NUM_MFCC)
#         zcr = librosa.feature.zero_crossing_rate(signal)
#         rms = librosa.feature.rms(y=signal)
#         spec_cent = librosa.feature.spectral_centroid(y=signal, sr=sr)
#         spec_bw = librosa.feature.spectral_bandwidth(y=signal, sr=sr)

#         # Create a list of features
#         features_list = [np.mean(mfccs[i]) for i in range(NUM_MFCC)]
#         features_list.append(np.mean(zcr))
#         features_list.append(np.mean(rms))
#         features_list.append(np.mean(spec_cent))
#         features_list.append(np.mean(spec_bw))
        
#         # --- The main change is here ---
#         # Create a list of feature names (the same ones used during training)
#         feature_names = [f'mfcc_{i+1}' for i in range(NUM_MFCC)] + ['zcr', 'rms', 'centroid', 'bandwidth']
        
#         # Return a Pandas DataFrame instead of a NumPy array
#         return pd.DataFrame([features_list], columns=feature_names)
        
#     except Exception as e:
#         messagebox.showerror("Error", f"Error processing audio: {e}")
#         return None

# # --- Function that runs on button click ---
# def analyze_audio():
#     if model is None or scaler is None:
#         messagebox.showerror("Error", "Model or scaler could not be loaded.")
#         return

#     # Open a dialog box to select a file
#     file_path = filedialog.askopenfilename(
#         title="Select Audio File",
#         filetypes=(("Audio Files", "*.wav *.mp3 *.ogg"), ("All files", "*.*"))
#     )
    
#     # If the user has selected a file...
#     if file_path:
#         # Update the result label to show processing status
#         result_label.config(text="Processing...", fg="blue")
#         root.update_idletasks() # To update the UI immediately

#         # Extract features from the selected audio file
#         features_df = extract_features(file_path) # This will return a DataFrame
        
#         if features_df is not None:
#             # Scale the features and make a prediction
#             features_scaled = scaler.transform(features_df)
#             prediction = model.predict(features_scaled)
#             prediction_proba = model.predict_proba(features_scaled)

#             # Determine the result text and color
#             if prediction[0] == 1:
#                 result_text = f"🚨 Scream Detected! (Confidence: {prediction_proba[0][1]*100:.2f}%)"
#                 result_color = "red"
#             else:
#                 result_text = f"✅ This is a Normal sound. (Confidence: {prediction_proba[0][0]*100:.2f}%)"
#                 result_color = "green"
            
#             # Show the final result on the label
#             result_label.config(text=result_text, fg=result_color)
#         else:
#             # Show an error if feature extraction failed
#             result_label.config(text="Could not process the file.", fg="red")

# # --- Main Tkinter Application ---

# # Create the main window
# root = tk.Tk()
# root.title("Human Scream Detection")
# root.geometry("500x250") # Window size

# # Create a frame to organize the widgets
# main_frame = tk.Frame(root, padx=20, pady=20)
# main_frame.pack(expand=True)

# # Label for instructions
# info_label = tk.Label(main_frame, text="Press the button below to analyze an audio file.", font=("Helvetica", 12))
# info_label.pack(pady=10)

# # Button
# analyze_button = tk.Button(main_frame, text="Select Audio File and Analyze", font=("Helvetica", 12, "bold"), command=analyze_audio, bg="#DDDDDD", width=35, height=2)
# analyze_button.pack(pady=20)

# # Label to show the result
# result_label = tk.Label(main_frame, text="", font=("Helvetica", 14, "bold"))
# result_label.pack(pady=10)

# # Start the application
# root.mainloop()
















# ---------------------------------------Enhanshment of UI-------------------------------------



# import tkinter as tk
# from tkinter import filedialog, messagebox, ttk
# import librosa
# import numpy as np
# import joblib
# import pandas as pd
# import threading
# import time

# # --- Load the saved model and scaler ---
# try:
#     model = joblib.load('logistic_regression_model.joblib')
#     scaler = joblib.load('scaler_logistic.joblib')
# except FileNotFoundError:
#     messagebox.showerror("Error", "Model or scaler file not found. Please run the saving script first.")
#     model = None
#     scaler = None

# # --- Function for audio processing and feature extraction ---
# def extract_features(file_path):
#     TARGET_SR = 22050
#     AUDIO_DURATION = 5
#     NUM_MFCC = 20
#     SAMPLES_PER_TRACK = TARGET_SR * AUDIO_DURATION
    
#     try:
#         signal, sr = librosa.load(file_path, sr=TARGET_SR, duration=AUDIO_DURATION)
#         if len(signal) < SAMPLES_PER_TRACK:
#             signal = np.pad(signal, (0, SAMPLES_PER_TRACK - len(signal)), 'constant')
        
#         mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=NUM_MFCC)
#         zcr = librosa.feature.zero_crossing_rate(signal)
#         rms = librosa.feature.rms(y=signal)
#         spec_cent = librosa.feature.spectral_centroid(y=signal, sr=sr)
#         spec_bw = librosa.feature.spectral_bandwidth(y=signal, sr=sr)

#         # Create feature list
#         features_list = [np.mean(mfccs[i]) for i in range(NUM_MFCC)]
#         features_list.append(np.mean(zcr))
#         features_list.append(np.mean(rms))
#         features_list.append(np.mean(spec_cent))
#         features_list.append(np.mean(spec_bw))
        
#         feature_names = [f'mfcc_{i+1}' for i in range(NUM_MFCC)] + ['zcr', 'rms', 'centroid', 'bandwidth']
#         return pd.DataFrame([features_list], columns=feature_names)
        
#     except Exception as e:
#         messagebox.showerror("Error", f"Error processing audio: {e}")
#         return None

# # --- Function to run analysis in background thread ---
# def analyze_audio():
#     if model is None or scaler is None:
#         messagebox.showerror("Error", "Model or scaler could not be loaded.")
#         return

#     file_path = filedialog.askopenfilename(
#         title="Select Audio File",
#         filetypes=(("Audio Files", "*.wav *.mp3 *.ogg"), ("All files", "*.*"))
#     )
    
#     if file_path:
#         result_label.config(text="🔄 Analyzing audio...", fg="blue")
#         progress_bar.start(10)
#         threading.Thread(target=process_audio, args=(file_path,)).start()

# def process_audio(file_path):
#     time.sleep(1)  # simulate loading time
#     features_df = extract_features(file_path)
    
#     if features_df is not None:
#         features_scaled = scaler.transform(features_df)
#         prediction = model.predict(features_scaled)
#         prediction_proba = model.predict_proba(features_scaled)

#         if prediction[0] == 1:
#             result_text = f"🚨 Scream Detected!\nConfidence: {prediction_proba[0][1]*100:.2f}%"
#             result_color = "white"
#             result_frame.config(bg="#e63946")
#         else:
#             result_text = f"✅ Normal Sound\nConfidence: {prediction_proba[0][0]*100:.2f}%"
#             result_color = "white"
#             result_frame.config(bg="#2a9d8f")
        
#         result_label.config(text=result_text, fg=result_color)
#     else:
#         result_label.config(text="❌ Could not process the file.", fg="red")
    
#     progress_bar.stop()

# # --- Main Tkinter Application ---
# root = tk.Tk()
# root.title("🔊 Human Scream Detection")
# root.geometry("600x400")
# root.configure(bg="#1d3557")

# # --- Header ---
# header = tk.Label(root, text="🔊 Human Scream Detection App", 
#                   font=("Helvetica", 20, "bold"), bg="#457b9d", fg="white", pady=15)
# header.pack(fill="x")

# # --- Instructions ---
# info_label = tk.Label(root, 
#     text="Select an audio file and the system will analyze it to detect if it's a scream.",
#     font=("Helvetica", 12), bg="#1d3557", fg="white", wraplength=500, justify="center")
# info_label.pack(pady=20)

# # --- Analyze Button ---
# analyze_button = tk.Button(root, text="🎵 Select Audio File and Analyze", 
#                            font=("Helvetica", 14, "bold"), command=analyze_audio, 
#                            bg="#f1faee", fg="#1d3557", activebackground="#a8dadc", 
#                            relief="flat", width=30, height=2, cursor="hand2")
# analyze_button.pack(pady=10)

# # --- Progress Bar ---
# progress_bar = ttk.Progressbar(root, mode="indeterminate", length=400)
# progress_bar.pack(pady=10)

# # --- Result Frame ---
# result_frame = tk.Frame(root, bg="#1d3557", bd=2, relief="groove")
# result_frame.pack(pady=20, ipadx=20, ipady=20, fill="x", padx=50)

# result_label = tk.Label(result_frame, text="", font=("Helvetica", 16, "bold"), bg=result_frame["bg"])
# result_label.pack()

# # --- Footer ---
# footer = tk.Label(root, text="Developed by Your Name 🔥", 
#                   font=("Helvetica", 10), bg="#1d3557", fg="lightgray")
# footer.pack(side="bottom", pady=10)

# # --- Run App ---
# root.mainloop()




# ------------------------------------claude enhamshmaent UI------------------------------------------

# import tkinter as tk
# from tkinter import filedialog, messagebox, ttk
# import librosa
# import numpy as np
# import joblib
# import pandas as pd
# from tkinter import font
# import time

# # --- Load the saved model and scaler ---
# try:
#     model = joblib.load('logistic_regression_model.joblib')
#     scaler = joblib.load('scaler_logistic.joblib')
# except FileNotFoundError:
#     messagebox.showerror("Error", "Model or scaler file not found. Please run the saving script first.")
#     model = None
#     scaler = None

# # --- Function for audio processing and feature extraction ---
# def extract_features(file_path):
#     TARGET_SR = 22050
#     AUDIO_DURATION = 5
#     NUM_MFCC = 20
#     SAMPLES_PER_TRACK = TARGET_SR * AUDIO_DURATION
    
#     try:
#         signal, sr = librosa.load(file_path, sr=TARGET_SR, duration=AUDIO_DURATION)
#         if len(signal) < SAMPLES_PER_TRACK:
#             signal = np.pad(signal, (0, SAMPLES_PER_TRACK - len(signal)), 'constant')
        
#         mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=NUM_MFCC)
#         zcr = librosa.feature.zero_crossing_rate(signal)
#         rms = librosa.feature.rms(y=signal)
#         spec_cent = librosa.feature.spectral_centroid(y=signal, sr=sr)
#         spec_bw = librosa.feature.spectral_bandwidth(y=signal, sr=sr)
        
#         features_list = [np.mean(mfccs[i]) for i in range(NUM_MFCC)]
#         features_list.append(np.mean(zcr))
#         features_list.append(np.mean(rms))
#         features_list.append(np.mean(spec_cent))
#         features_list.append(np.mean(spec_bw))
        
#         feature_names = [f'mfcc_{i+1}' for i in range(NUM_MFCC)] + ['zcr', 'rms', 'centroid', 'bandwidth']
#         return pd.DataFrame([features_list], columns=feature_names)
#     except Exception as e:
#         messagebox.showerror("Error", f"Error processing audio: {e}")
#         return None

# # --- Function that runs on button click ---
# def analyze_audio():
#     if model is None or scaler is None:
#         messagebox.showerror("Error", "Model or scaler could not be loaded.")
#         return
    
#     file_path = filedialog.askopenfilename(
#         title="Select Audio File",
#         filetypes=(("Audio Files", "*.wav *.mp3 *.ogg"), ("All files", "*.*"))
#     )
    
#     if file_path:
#         # Update UI to show processing
#         result_frame.config(bg="#FFF3CD")  # Light yellow background
#         result_icon.config(text="🔄", fg="#856404")
#         result_text.config(text="Processing audio file...", fg="#856404")
#         confidence_label.config(text="")
#         progress_bar.start(10)
#         root.update()  # Force UI update
        
#         # Process in small chunks to keep UI responsive
#         try:
#             # Stage 1: Loading audio file
#             result_text.config(text="Loading audio file...", fg="#856404")
#             root.update()
#             time.sleep(1.0)  # 1 second delay
            
#             # Extract features
#             result_text.config(text="Extracting audio features...", fg="#856404")
#             root.update()
#             features_df = extract_features(file_path)
#             time.sleep(1.5)  # 1.5 second delay
#             root.update()
            
#             if features_df is not None:
#                 # Stage 2: Preprocessing
#                 result_text.config(text="Preprocessing features...", fg="#856404")
#                 root.update()
#                 time.sleep(1.0)  # 1 second delay
                
#                 features_scaled = scaler.transform(features_df)
#                 root.update()
                
#                 # Stage 3: Making prediction
#                 result_text.config(text="Analyzing with AI model...", fg="#856404")
#                 root.update()
#                 time.sleep(1.5)  # 1.5 second delay
                
#                 prediction = model.predict(features_scaled)
#                 prediction_proba = model.predict_proba(features_scaled)
#                 root.update()
                
#                 # Stop progress bar
#                 progress_bar.stop()
            
#                 # Update results based on prediction
#                 if prediction[0] == 1:
#                     # Scream detected
#                     result_frame.config(bg="#F8D7DA")  # Light red
#                     result_icon.config(text="🚨", fg="#721C24")
#                     result_text.config(text="SCREAM DETECTED!", fg="#721C24")
#                     confidence_label.config(text=f"Confidence: {prediction_proba[0][1]*100:.1f}%", fg="#721C24")
#                 else:
#                     # Normal sound
#                     result_frame.config(bg="#D4EDDA")  # Light green
#                     result_icon.config(text="✅", fg="#155724")
#                     result_text.config(text="Normal Sound", fg="#155724")
#                     confidence_label.config(text=f"Confidence: {prediction_proba[0][0]*100:.1f}%", fg="#155724")
#             else:
#                 # Error processing file
#                 progress_bar.stop()
#                 result_frame.config(bg="#F8D7DA")
#                 result_icon.config(text="❌", fg="#721C24")
#                 result_text.config(text="Error processing file", fg="#721C24")
#                 confidence_label.config(text="Please try a different audio file", fg="#721C24")
                
#         except Exception as e:
#             # Handle any unexpected errors
#             progress_bar.stop()
#             result_frame.config(bg="#F8D7DA")
#             result_icon.config(text="❌", fg="#721C24")
#             result_text.config(text="Processing Error", fg="#721C24")
#             confidence_label.config(text=f"Error: {str(e)}", fg="#721C24")

# # --- Create the main window ---
# root = tk.Tk()
# root.title("🎵 Human Scream Detection AI")
# root.geometry("600x500")
# root.configure(bg="#F8F9FA")
# root.resizable(False, False)

# # Create custom fonts
# title_font = font.Font(family="Helvetica", size=18, weight="bold")
# subtitle_font = font.Font(family="Helvetica", size=11)
# button_font = font.Font(family="Helvetica", size=12, weight="bold")
# result_font = font.Font(family="Helvetica", size=16, weight="bold")

# # --- Header Section ---
# header_frame = tk.Frame(root, bg="#343A40", height=100)
# header_frame.pack(fill="x")
# header_frame.pack_propagate(False)

# title_label = tk.Label(
#     header_frame, 
#     text="🎵 Human Scream Detection AI",
#     font=title_font,
#     fg="white",
#     bg="#343A40"
# )
# title_label.pack(pady=(15, 5))

# subtitle_label = tk.Label(
#     header_frame,
#     text="Advanced ML-powered audio analysis for scream detection",
#     font=subtitle_font,
#     fg="#ADB5BD",
#     bg="#343A40"
# )
# subtitle_label.pack()

# # --- Main Content Area ---
# main_frame = tk.Frame(root, bg="#F8F9FA", padx=40, pady=30)
# main_frame.pack(expand=True, fill="both")

# # Instructions
# instructions_frame = tk.Frame(main_frame, bg="#E9ECEF", relief="solid", bd=1)
# instructions_frame.pack(fill="x", pady=(0, 20))

# instructions_label = tk.Label(
#     instructions_frame,
#     text="📁 Select an audio file (.wav, .mp3, .ogg) to analyze",
#     font=("Helvetica", 11),
#     fg="#495057",
#     bg="#E9ECEF",
#     pady=15
# )
# instructions_label.pack()

# # Analyze Button with hover effects
# button_frame = tk.Frame(main_frame, bg="#F8F9FA")
# button_frame.pack(pady=10)

# analyze_button = tk.Button(
#     button_frame,
#     text="🔍 Select & Analyze Audio File",
#     font=button_font,
#     command=analyze_audio,
#     bg="#007BFF",
#     fg="white",
#     activebackground="#0056B3",
#     activeforeground="white",
#     relief="flat",
#     width=25,
#     height=2,
#     cursor="hand2"
# )
# analyze_button.pack()

# # Progress Bar
# progress_bar = ttk.Progressbar(
#     main_frame,
#     mode='indeterminate',
#     length=300,
#     style="TProgressbar"
# )
# progress_bar.pack(pady=10)

# # --- Results Section ---
# result_frame = tk.Frame(main_frame, bg="#F8F9FA", relief="solid", bd=2, padx=20, pady=20)
# result_frame.pack(fill="x", pady=20)

# result_icon = tk.Label(
#     result_frame,
#     text="🎯",
#     font=("Helvetica", 24),
#     bg="#F8F9FA"
# )
# result_icon.pack()

# result_text = tk.Label(
#     result_frame,
#     text="Ready to analyze audio",
#     font=result_font,
#     bg="#F8F9FA",
#     fg="#6C757D"
# )
# result_text.pack(pady=(5, 0))

# confidence_label = tk.Label(
#     result_frame,
#     text="Upload an audio file to get started",
#     font=("Helvetica", 10),
#     bg="#F8F9FA",
#     fg="#6C757D"
# )
# confidence_label.pack(pady=(5, 0))

# # --- Footer ---
# footer_frame = tk.Frame(root, bg="#343A40", height=40)
# footer_frame.pack(fill="x", side="bottom")
# footer_frame.pack_propagate(False)

# footer_label = tk.Label(
#     footer_frame,
#     text="Powered by Machine Learning • Built with Python & Tkinter",
#     font=("Helvetica", 9),
#     fg="#ADB5BD",
#     bg="#343A40"
# )
# footer_label.pack(expand=True)

# # --- Hover effects for button ---
# def on_enter(e):
#     analyze_button.config(bg="#0056B3")

# def on_leave(e):
#     analyze_button.config(bg="#007BFF")

# analyze_button.bind("<Enter>", on_enter)
# analyze_button.bind("<Leave>", on_leave)

# # Configure ttk style for progress bar
# style = ttk.Style()
# style.theme_use('clam')
# style.configure("TProgressbar", background='#007BFF', troughcolor='#E9ECEF', borderwidth=0, lightcolor='#007BFF', darkcolor='#007BFF')

# # Start the application
# root.mainloop()



# -------------------------------------Matirial theam UI------------------------------------

# import tkinter as tk
# from tkinter import filedialog, messagebox, ttk
# import librosa
# import numpy as np
# import joblib
# import pandas as pd
# from tkinter import font
# import time
# import threading

# # --- Load the saved model and scaler ---
# try:
#     model = joblib.load('logistic_regression_model.joblib')
#     scaler = joblib.load('scaler_logistic.joblib')
# except FileNotFoundError:
#     messagebox.showerror("Error", "Model or scaler file not found. Please run the saving script first.")
#     model = None
#     scaler = None

# # --- Function for audio processing and feature extraction ---
# def extract_features(file_path):
#     TARGET_SR = 22050
#     AUDIO_DURATION = 5
#     NUM_MFCC = 20
#     SAMPLES_PER_TRACK = TARGET_SR * AUDIO_DURATION
    
#     try:
#         signal, sr = librosa.load(file_path, sr=TARGET_SR, duration=AUDIO_DURATION)
#         if len(signal) < SAMPLES_PER_TRACK:
#             signal = np.pad(signal, (0, SAMPLES_PER_TRACK - len(signal)), 'constant')
        
#         mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=NUM_MFCC)
#         zcr = librosa.feature.zero_crossing_rate(signal)
#         rms = librosa.feature.rms(y=signal)
#         spec_cent = librosa.feature.spectral_centroid(y=signal, sr=sr)
#         spec_bw = librosa.feature.spectral_bandwidth(y=signal, sr=sr)
        
#         features_list = [np.mean(mfccs[i]) for i in range(NUM_MFCC)]
#         features_list.append(np.mean(zcr))
#         features_list.append(np.mean(rms))
#         features_list.append(np.mean(spec_cent))
#         features_list.append(np.mean(spec_bw))
        
#         feature_names = [f'mfcc_{i+1}' for i in range(NUM_MFCC)] + ['zcr', 'rms', 'centroid', 'bandwidth']
#         return pd.DataFrame([features_list], columns=feature_names)
#     except Exception as e:
#         messagebox.showerror("Error", f"Error processing audio: {e}")
#         return None

# # --- Function triggered when button is clicked ---
# def analyze_audio():
#     if model is None or scaler is None:
#         messagebox.showerror("Error", "Model or scaler could not be loaded.")
#         return
    
#     file_path = filedialog.askopenfilename(
#         title="Select Audio File",
#         filetypes=(("Audio Files", "*.wav *.mp3 *.ogg"), ("All files", "*.*"))
#     )
    
#     if file_path:
#         # Reset UI to "processing" state
#         result_frame.config(bg="#FFF3CD")
#         result_icon.config(text="🔄", fg="#856404")
#         result_text.config(text="Processing audio file...", fg="#856404")
#         confidence_label.config(text="")
#         progress_bar.start(10)
#         root.update()
        
#         # Run processing in background thread
#         threading.Thread(target=process_file, args=(file_path,), daemon=True).start()

# # --- Actual audio processing logic ---
# def process_file(file_path):
#     try:
#         # Stage 1: Loading audio file
#         result_text.config(text="Loading audio file...", fg="#856404")
#         time.sleep(1.0)
        
#         # Extract features
#         result_text.config(text="Extracting audio features...", fg="#856404")
#         features_df = extract_features(file_path)
#         time.sleep(1.0)
        
#         if features_df is not None:
#             result_text.config(text="Preprocessing features...", fg="#856404")
#             time.sleep(0.8)
#             features_scaled = scaler.transform(features_df)
            
#             result_text.config(text="Analyzing with AI model...", fg="#856404")
#             time.sleep(1.0)
#             prediction = model.predict(features_scaled)
#             prediction_proba = model.predict_proba(features_scaled)
            
#             # Stop progress bar
#             progress_bar.stop()
            
#             # Update results
#             if prediction[0] == 1:
#                 result_frame.config(bg="#F8D7DA")
#                 result_icon.config(text="🚨", fg="#721C24")
#                 result_text.config(text="SCREAM DETECTED!", fg="#721C24")
#                 confidence_label.config(text=f"Confidence: {prediction_proba[0][1]*100:.1f}%", fg="#721C24")
#             else:
#                 result_frame.config(bg="#D4EDDA")
#                 result_icon.config(text="✅", fg="#155724")
#                 result_text.config(text="Normal Sound", fg="#155724")
#                 confidence_label.config(text=f"Confidence: {prediction_proba[0][0]*100:.1f}%", fg="#155724")
#         else:
#             progress_bar.stop()
#             result_frame.config(bg="#F8D7DA")
#             result_icon.config(text="❌", fg="#721C24")
#             result_text.config(text="Error processing file", fg="#721C24")
#             confidence_label.config(text="Please try a different audio file", fg="#721C24")
            
#     except Exception as e:
#         progress_bar.stop()
#         result_frame.config(bg="#F8D7DA")
#         result_icon.config(text="❌", fg="#721C24")
#         result_text.config(text="Processing Error", fg="#721C24")
#         confidence_label.config(text=f"Error: {str(e)}", fg="#721C24")

# # --- Create the main window ---
# root = tk.Tk()
# root.title("🎵 Human Scream Detection AI")
# root.geometry("600x500")
# root.configure(bg="#F8F9FA")
# root.resizable(False, False)

# # Custom fonts
# title_font = font.Font(family="Helvetica", size=18, weight="bold")
# subtitle_font = font.Font(family="Helvetica", size=11)
# button_font = font.Font(family="Helvetica", size=12, weight="bold")
# result_font = font.Font(family="Helvetica", size=16, weight="bold")

# # --- Header ---
# header_frame = tk.Frame(root, bg="#343A40", height=100)
# header_frame.pack(fill="x")
# header_frame.pack_propagate(False)

# title_label = tk.Label(header_frame, text="🎵 Human Scream Detection AI",
#                        font=title_font, fg="white", bg="#343A40")
# title_label.pack(pady=(15, 5))

# subtitle_label = tk.Label(header_frame,
#     text="Advanced ML-powered audio analysis for scream detection",
#     font=subtitle_font, fg="#ADB5BD", bg="#343A40")
# subtitle_label.pack()

# # --- Main Content ---
# main_frame = tk.Frame(root, bg="#F8F9FA", padx=40, pady=30)
# main_frame.pack(expand=True, fill="both")

# instructions_frame = tk.Frame(main_frame, bg="#E9ECEF", relief="solid", bd=1)
# instructions_frame.pack(fill="x", pady=(0, 20))

# instructions_label = tk.Label(instructions_frame,
#     text="📁 Select an audio file (.wav, .mp3, .ogg) to analyze",
#     font=("Helvetica", 11), fg="#495057", bg="#E9ECEF", pady=15)
# instructions_label.pack()

# # Analyze Button
# button_frame = tk.Frame(main_frame, bg="#F8F9FA")
# button_frame.pack(pady=10)

# analyze_button = tk.Button(button_frame, text="🔍 Select & Analyze Audio File",
#                            font=button_font, command=analyze_audio,
#                            bg="#007BFF", fg="white",
#                            activebackground="#0056B3", activeforeground="white",
#                            relief="flat", width=25, height=2, cursor="hand2")
# analyze_button.pack()

# # Progress Bar
# progress_bar = ttk.Progressbar(main_frame, mode='indeterminate',
#                                length=300, style="TProgressbar")
# progress_bar.pack(pady=10)

# # --- Results Section ---
# result_frame = tk.Frame(main_frame, bg="#F8F9FA", relief="solid", bd=2, padx=20, pady=20)
# result_frame.pack(fill="x", pady=20)

# result_icon = tk.Label(result_frame, text="🎯", font=("Helvetica", 24), bg="#F8F9FA")
# result_icon.pack()

# result_text = tk.Label(result_frame, text="Ready to analyze audio",
#                        font=result_font, bg="#F8F9FA", fg="#6C757D")
# result_text.pack(pady=(5, 0))

# confidence_label = tk.Label(result_frame, text="Upload an audio file to get started",
#                             font=("Helvetica", 10), bg="#F8F9FA", fg="#6C757D")
# confidence_label.pack(pady=(5, 0))

# # --- Footer ---
# footer_frame = tk.Frame(root, bg="#343A40", height=40)
# footer_frame.pack(fill="x", side="bottom")
# footer_frame.pack_propagate(False)

# footer_label = tk.Label(footer_frame,
#     text="Powered by Machine Learning • Built with Python & Tkinter",
#     font=("Helvetica", 9), fg="#ADB5BD", bg="#343A40")
# footer_label.pack(expand=True)

# # --- Hover Effects ---
# def on_enter(e): analyze_button.config(bg="#0056B3")
# def on_leave(e): analyze_button.config(bg="#007BFF")
# analyze_button.bind("<Enter>", on_enter)
# analyze_button.bind("<Leave>", on_leave)

# # --- Progress Bar Style ---
# style = ttk.Style()
# style.theme_use('clam')
# style.configure("TProgressbar", background='#007BFF',
#                 troughcolor='#E9ECEF', borderwidth=0,
#                 lightcolor='#007BFF', darkcolor='#007BFF')

# # Start App
# root.mainloop()



#  ------------------------------------Add Player in to UI-------------------------

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import librosa
import numpy as np
import joblib
import pandas as pd
from tkinter import font
import time
import threading
import pygame

# --- Initialize Pygame Mixer ---
pygame.mixer.init()

# --- Global variables ---
current_file_path = None
current_filename = None

# --- Load the saved model and scaler ---
try:
    model = joblib.load('logistic_regression_model.joblib')
    scaler = joblib.load('scaler_logistic.joblib')
except FileNotFoundError:
    messagebox.showerror("Error", "Model or scaler file not found. Please run the saving script first.")
    model = None
    scaler = None

# --- Function for audio processing and feature extraction ---
def extract_features(file_path):
    TARGET_SR = 22050
    AUDIO_DURATION = 5
    NUM_MFCC = 20
    SAMPLES_PER_TRACK = TARGET_SR * AUDIO_DURATION
    try:
        signal, sr = librosa.load(file_path, sr=TARGET_SR, duration=AUDIO_DURATION)
        if len(signal) < SAMPLES_PER_TRACK:
            signal = np.pad(signal, (0, SAMPLES_PER_TRACK - len(signal)), 'constant')
        mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=NUM_MFCC)
        zcr = librosa.feature.zero_crossing_rate(signal)
        rms = librosa.feature.rms(y=signal)
        spec_cent = librosa.feature.spectral_centroid(y=signal, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=signal, sr=sr)
        features_list = [np.mean(mfccs[i]) for i in range(NUM_MFCC)]
        features_list.extend([np.mean(zcr), np.mean(rms), np.mean(spec_cent), np.mean(spec_bw)])
        feature_names = [f'mfcc_{i+1}' for i in range(NUM_MFCC)] + ['zcr', 'rms', 'centroid', 'bandwidth']
        return pd.DataFrame([features_list], columns=feature_names)
    except Exception as e:
        messagebox.showerror("Error", f"Error processing audio: {e}")
        return None

# --- Function to play or stop the audio ---
def toggle_playback():
    global current_file_path
    if current_file_path:
        try:
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
                play_button.config(text="▶️ Play Sound")
            else:
                pygame.mixer.music.load(current_file_path)
                pygame.mixer.music.play()
                play_button.config(text="⏹️ Stop Sound")
        except pygame.error as e:
            messagebox.showerror("Playback Error", f"Could not play the file. Error: {e}")

# --- Function to handle file selection only ---
def select_audio_file():
    global current_file_path, current_filename
    
    file_path = filedialog.askopenfilename(
        title="Select Audio File",
        filetypes=(("Audio Files", "*.wav *.mp3 *.ogg"), ("All files", "*.*"))
    )
    if file_path:
        current_file_path = file_path
        current_filename = file_path.split('/')[-1]
        
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
        
        filename_label.config(text=f"Selected: {current_filename}")
        play_button.config(text="▶️ Play Sound")
        
        selection_details_frame.pack(pady=10)
        
        # Reset the result display
        result_frame.config(bg="#F8F9FA")
        result_icon.config(text="🎯", bg="#F8F9FA")
        result_text.config(text="Ready to analyze audio", bg="#F8F9FA", fg="#6C757D")

# --- Function to start the analysis process ---
def start_analysis():
    if model is None or scaler is None:
        messagebox.showerror("Error", "Model or scaler could not be loaded.")
        return
    if current_file_path is None:
        messagebox.showwarning("No File", "Please select an audio file first.")
        return

    play_button.config(state="disabled")
    analyze_button.config(state="disabled")
    
    result_frame.config(bg="#FFF3CD")
    result_icon.config(text="🔄", fg="#856404", bg="#FFF3CD")
    result_text.config(text="Processing audio file...", fg="#856404", bg="#FFF3CD")
    
    progress_bar.pack(pady=15)
    progress_bar.start(10)
    
    root.update()
    
    threading.Thread(target=process_file, args=(current_file_path,), daemon=True).start()

# --- Actual audio processing logic ---
def process_file(file_path):
    try:
        result_text.config(text="Extracting audio features...")
        features_df = extract_features(file_path)
        time.sleep(1.0)
        
        if features_df is not None:
            result_text.config(text="Preprocessing features...")
            time.sleep(0.8)
            features_scaled = scaler.transform(features_df)
            
            result_text.config(text="Analyzing with AI model...")
            time.sleep(1.0)
            prediction = model.predict(features_scaled)
            
            progress_bar.stop()
            progress_bar.pack_forget()
            
            if prediction[0] == 1:
                result_frame.config(bg="#F8D7DA")
                result_icon.config(text="🚨", fg="#721C24", bg="#F8D7DA")
                result_text.config(text="SCREAM DETECTED!", fg="#721C24", bg="#F8D7DA")
            else:
                result_frame.config(bg="#D4EDDA")
                result_icon.config(text="✅", fg="#155724", bg="#D4EDDA")
                result_text.config(text="Normal Sound", fg="#155724", bg="#D4EDDA")
        else:
            progress_bar.stop()
            progress_bar.pack_forget()
            result_frame.config(bg="#F8D7DA")
            result_icon.config(text="❌", fg="#721C24", bg="#F8D7DA")
            result_text.config(text="Error processing file", fg="#721C24", bg="#F8D7DA")
    except Exception as e:
        progress_bar.stop()
        progress_bar.pack_forget()
        result_frame.config(bg="#F8D7DA")
        result_icon.config(text="❌", fg="#721C24", bg="#F8D7DA")
        result_text.config(text="Processing Error", fg="#721C24", bg="#F8D7DA")
    finally:
        play_button.config(state="normal")
        analyze_button.config(state="normal")

# --- Create the main window ---
root = tk.Tk()
root.title("🎵 Human Scream Detection AI")
root.geometry("600x500") # Slightly reduced height
root.configure(bg="#F8F9FA")
root.resizable(False, False)

# --- Fonts ---
title_font = font.Font(family="Helvetica", size=18, weight="bold")
subtitle_font = font.Font(family="Helvetica", size=11)
button_font = font.Font(family="Helvetica", size=12, weight="bold")
result_font = font.Font(family="Helvetica", size=16, weight="bold")
action_button_font = font.Font(family="Helvetica", size=10, weight="bold")

# --- Header ---
header_frame = tk.Frame(root, bg="#343A40", height=100)
header_frame.pack(fill="x", side="top")
header_frame.pack_propagate(False)
tk.Label(header_frame, text="🎵 Human Scream Detection AI", font=title_font, fg="white", bg="#343A40").pack(pady=(15, 5))
tk.Label(header_frame, text="Advanced ML-powered audio analysis", font=subtitle_font, fg="#ADB5BD", bg="#343A40").pack()

# --- Main Content Frame ---
main_frame = tk.Frame(root, bg="#F8F9FA", padx=40, pady=20)
main_frame.pack(expand=True, fill="both")

# --- Step 1: File Selection ---
select_button_frame = tk.Frame(main_frame, bg="#F8F9FA")
select_button_frame.pack(pady=10)
select_button = tk.Button(select_button_frame, text="📂 Select Audio File", font=button_font, command=select_audio_file, bg="#007BFF", fg="white", activebackground="#0056B3", activeforeground="white", relief="flat", width=25, height=2, cursor="hand2")
select_button.pack()

# --- Step 2: Controls for Selected File (Initially Hidden) ---
selection_details_frame = tk.Frame(main_frame, bg="#F8F9FA")
filename_label = tk.Label(selection_details_frame, text="", font=("Helvetica", 11, "italic"), fg="#495057", bg="#F8F9FA")
filename_label.pack(pady=(10, 5))
control_frame = tk.Frame(selection_details_frame, bg="#F8F9FA")
control_frame.pack(pady=5)
play_button = tk.Button(control_frame, text="▶️ Play Sound", font=action_button_font, command=toggle_playback, bg="#28A745", fg="white", activebackground="#218838", width=15, relief="flat", cursor="hand2")
play_button.pack(side="left", padx=5)
analyze_button = tk.Button(control_frame, text="🔬 Analyze Now", font=action_button_font, command=start_analysis, bg="#17A2B8", fg="white", activebackground="#138496", width=15, relief="flat", cursor="hand2")
analyze_button.pack(side="left", padx=5)

# --- Progress Bar ---
progress_bar = ttk.Progressbar(main_frame, mode='indeterminate', length=300)

# --- Step 3: Results Section ---
result_frame = tk.Frame(main_frame, bg="#F8F9FA", relief="solid", bd=2, padx=20, pady=20)
result_frame.pack(fill="x", pady=10)
result_icon = tk.Label(result_frame, text="🎯", font=("Helvetica", 24), bg="#F8F9FA")
result_icon.pack()
result_text = tk.Label(result_frame, text="Ready to analyze audio", font=result_font, bg="#F8F9FA", fg="#6C757D")
result_text.pack(pady=5) # Adjusted padding

# --- Footer ---
footer_frame = tk.Frame(root, bg="#343A40", height=40)
footer_frame.pack(fill="x", side="bottom")
footer_frame.pack_propagate(False)
tk.Label(footer_frame, text="Powered by Machine Learning • Trupesh", font=("Helvetica", 9), fg="#ADB5BD", bg="#343A40").pack(expand=True)

# --- Style and Hover Effects ---
select_button.bind("<Enter>", lambda e: select_button.config(bg="#0056B3"))
select_button.bind("<Leave>", lambda e: select_button.config(bg="#007BFF"))
style = ttk.Style()
style.theme_use('clam')
style.configure("TProgressbar", background='#007BFF', troughcolor='#E9ECEF', borderwidth=0)

# --- Start App ---
root.mainloop()
pygame.mixer.quit()