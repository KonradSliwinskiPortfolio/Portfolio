from ultralytics import YOLO

model = YOLO('runs/classify/train6/weights/best.pt') 

plazma_do_spawania = 0
plazma_do_napawania = 0
TIG = 0
GMA = 0
GMA_2 = 0

for i in range(1, 51):
    results = model(f"{i}.jpg")
    probs = results[0].probs  

    probs_values = probs.data.tolist()

    probs0, probs1, probs2, probs3, probs4 = probs_values

    if probs0 >= probs1 and probs0 >= probs2 and probs0 >= probs3 and probs0 >= probs4:
        plazma_do_spawania += 1
    elif probs1 >= probs0 and probs1 >= probs2 and probs1 >= probs3 and probs1 >= probs4:
        plazma_do_napawania += 1
    elif probs2 >= probs0 and probs2 >= probs1 and probs2 >= probs3 and probs2 >= probs4:
        TIG += 1
    elif probs3 >= probs0 and probs3 >= probs1 and probs3 >= probs2 and probs3 >= probs4:
        GMA += 1
    else:
        GMA_2 += 1

print(f'plazma_do_spawania= {plazma_do_spawania}')
print(f'plazma_do_napawania= {plazma_do_napawania}')
print(f'TIG= {TIG}')
print(f'GMA= {GMA}')
print(f'GMA_2= {GMA_2}')
