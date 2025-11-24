print("=== ข้อมูลผลไม้ในแต่ละกล่อง ===")
a1 = int(input("กล่อง 1: แอปเปิล = "))
b1 = int(input("กล่อง 1: กล้วย = "))

a2 = int(input("กล่อง 2: แอปเปิล = "))
b2 = int(input("กล่อง 2: กล้วย = "))

p1 = 0.5
p2 = 0.5

pA_from_box1 = a1 / (a1 + b1)
pA_from_box2 = a2 / (a2 + b2)

pA = p1 * pA_from_box1 + p2 * pA_from_box2

p_box1_given_A = (pA_from_box1 * p1) / pA
p_box2_given_A = 1 - p_box1_given_A

print("\n=== ผลลัพธ์ ===")
print("โอกาสหยิบได้แอปเปิล = {:.3f}".format(pA))
print("ถ้าหยิบได้แอปเปิล มาจากกล่อง 1 = {:.3f}".format(p_box1_given_A))
print("ถ้าหยิบได้แอปเปิล มาจากกล่อง 2 = {:.3f}".format(p_box2_given_A))
