import torch 

a = torch.load('a.pt')
b = torch.load('b.pt')
truth = torch.matmul(a, b)

lts = torch.load('lts.pt')

torch.testing.assert_close(truth, lts)
print("Truth and LTS values match!")

cpp_output = torch.load('cpp_outs.pt')
torch.testing.assert_close(truth, cpp_output)
print("Truth and CPP values match!")

torch.testing.assert_close(lts, cpp_output)
print("LTS and CPP match! (expected b/c the previous two cases passed)")