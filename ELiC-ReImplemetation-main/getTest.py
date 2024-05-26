import re

def extract_test_info(log_file):
    test_info = []
    cnt = 0
    with open(log_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if "Test epoch" in line:
                match = re.search(r'Bpp loss: (\d+\.\d+) \|.*PSNR: (\d+\.\d+)', line)
                if match:
                    cnt += 1
                    bpp_loss = float(match.group(1))
                    psnr = float(match.group(2))
                    test_info.append((cnt, bpp_loss, psnr))
    return test_info

log_file = 'test.log'
test_info = extract_test_info(log_file)

with open('test.txt', 'w') as f:
    f.write("Test Information\n")

# 打印提取的信息
for cnt, bpp_loss, psnr in test_info:
    print(f"Epoch: {cnt}, BPP Loss: {bpp_loss}, PSNR: {psnr}")
    with open('test.txt', 'a') as f:
        f.write(f"Epoch: {cnt}, BPP Loss: {bpp_loss}, PSNR: {psnr}\n")
