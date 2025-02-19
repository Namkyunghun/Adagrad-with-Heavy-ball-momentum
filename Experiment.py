import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

k = 500               
num_iterations = 60000  
lr = 0.004            
beta = 0.004          
epsilon = 1e-6        

dimensions = [5000, 10000, 50000, 100000, 500000]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def custom_x_formatter(x, pos):
    if x < 10000:
        return f"{int(x)}"
    else:
        return f"{int(x):,}"


def format_dimension(d):
    if d < 10000:
        return f"{d}"
    else:
        return f"{d:,}"
    
plt.figure(figsize=(10, 6))

for d in dimensions:
    torch.manual_seed(42)
    
    A = torch.randn(k, d, device=device)
    A = A / A.norm(dim=1, keepdim=True)

    x = torch.zeros(d, device=device, requires_grad=True)
    
    
    acc = torch.zeros_like(x)
    m = torch.zeros_like(x)
    
    grad_norms = []
    iterations_list = []
    for t in range(num_iterations):
        ax = A.matmul(x)      
        loss = torch.mean(F.softplus(-ax))
        
        loss.backward()

        grad_norm_sq = (x.grad ** 2).sum().item()
        grad_norms.append(grad_norm_sq)
        iterations_list.append(t)
        
        with torch.no_grad():
            g = x.grad 
            m = (1 - beta) * m + beta * g          
            acc += g ** 2 
            x -= lr * m / (torch.sqrt(acc) + epsilon)
            x.grad.zero_()
            

    plt.plot(iterations_list, grad_norms, label=f'd = {format_dimension(d)}')

plt.xlabel('Iteration Number')
plt.ylabel(r'Gradient Norm Squared $\|\nabla f(x)\|_2^2$')
plt.title('Gradient Norm Squared vs. Iteration for Different Dimensions')
plt.legend()
plt.grid(True)
plt.tight_layout()


plt.gca().xaxis.set_major_formatter(mticker.FuncFormatter(custom_x_formatter))


plt.savefig('gradient_norm_squared.png')
plt.savefig('gradient_norm_squared.pdf')
plt.show()