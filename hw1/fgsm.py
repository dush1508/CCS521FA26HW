import torch
import torch.nn as nn


# fix seed so that random initialization always performs the same 
torch.manual_seed(13)


# create the model N as described in the question
N = nn.Sequential(nn.Linear(10, 10, bias=False),
                  nn.ReLU(),
                  nn.Linear(10, 10, bias=False),
                  nn.ReLU(),
                  nn.Linear(10, 3, bias=False))

# random input
x = torch.rand((1,10)) # the first dimension is the batch size; the following dimensions the actual dimension of the data
x.requires_grad_() # this is required so we can compute the gradient w.r.t x

t = 0 # target class

epsReal = 0.5  #depending on your data this might be large or small
eps = epsReal - 1e-7 # small constant to offset floating-point erros

# The network N classfies x as belonging to class 2
original_class =  N(x).argmax(dim=1).item()  # TO LEARN: make sure you understand this expression
print("Original Class: ", original_class)
assert(original_class == 2)

# compute gradient
# note that CrossEntropyLoss() combines the cross-entropy loss and an implicit softmax function
L = nn.CrossEntropyLoss()
loss = L(N(x), torch.tensor([t], dtype=torch.long)) # TO LEARN: make sure you understand this line
loss.backward()

# your code here
# adv_x should be computed from x according to the fgsm-style perturbation such that the new class of xBar is the target class t above
# hint: you can compute the gradient of the loss w.r.t to x as x.grad
# From slide 15 lol
adv_x = x.detach() - eps * x.grad.sign()

new_class = N(adv_x).argmax(dim=1).item()
print("New Class: ", new_class)
assert(new_class == t)
# it is not enough that adv_x is classified as t. We also need to make sure it is 'close' to the original x. 
print(torch.norm((x-adv_x),  p=float('inf')).data)
assert( torch.norm((x-adv_x), p=float('inf')) <= epsReal)


# Part B: goal is to target class 1
t = 1
target = torch.tensor([t], dtype=torch.long)
x_original = x.detach()

# Trying a one-step FGSM attack.
loss = L(N(x), target)
grad, = torch.autograd.grad(loss, x)
adv_x = x_original - eps * grad.sign()
print("\nPart B: FGSM class:", N(adv_x).argmax(dim=1).item())

# We will move straight to PGD 
torch.manual_seed(42)
alpha = 0.01
for attempt in range(40):

    # Starting with a random point within the ball
    adv_x = x_original + torch.empty_like(x).uniform_(-eps, eps)

    for step in range(300):
        adv_x.requires_grad_()
        loss = L(N(adv_x), target)
        grad, = torch.autograd.grad(loss, adv_x)

        #same as targeted iterative fgsm
        adv_x = adv_x.detach() - alpha * grad.sign()
        adv_x = x_original + (adv_x - x_original).clamp(-eps, eps)

        if N(adv_x).argmax(dim=1).item() == t:
            break

    if N(adv_x).argmax(dim=1).item() == t:
        print("Wow! Target class found! in .....", attempt, "attempts")
        break

new_class = N(adv_x).argmax(dim=1).item()
distance = (adv_x - x_original).abs().max().item()
print("Method: targeted PGD")
print("Budget (e ball):", epsReal, "Step size (alpha):", alpha)
print("PGD class:", new_class)
print("L_inf distance:", distance)
assert new_class == t
assert distance <= epsReal

print("Let's go!")

# Make this true if you want to indulge my curiosities
# I want to see what is the average attempt needed to find the target class when doing PGD
if True:
    attempts = []
    for _ in range(100):
        alpha = 0.01
        for attempt in range(40):

            # Starting with a random point within the ball
            adv_x = x_original + torch.empty_like(x).uniform_(-eps, eps)

            for step in range(300):
                adv_x.requires_grad_()
                loss = L(N(adv_x), target)
                grad, = torch.autograd.grad(loss, adv_x)

                #same as targeted iterative fgsm
                adv_x = adv_x.detach() - alpha * grad.sign()
                adv_x = x_original + (adv_x - x_original).clamp(-eps, eps)

                if N(adv_x).argmax(dim=1).item() == t:
                    break

            if N(adv_x).argmax(dim=1).item() == t:
                attempts.append(attempt)
                break
    print("Average attempts needed:", sum(attempts) / len(attempts))