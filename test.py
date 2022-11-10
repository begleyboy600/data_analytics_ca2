num = 1.1010587094603724e-05

print(f"{num:.10f}")

ans = 1 - 0.0000110106
print(ans)

pkl_filename = "uni_prediction_model.pkl"
"""
# save model to pickle file
with open(pkl_filename, 'wb') as file:
    pickle.dump(model6, file)
"""

# open pickle file
with open(pkl_filename, 'rb') as file:
    model6 = pickle.load(file)