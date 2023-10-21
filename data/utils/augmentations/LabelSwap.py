import torch

class LabelSwap:
    def __init__(self, group: int):
        self.group = group
        
        self.swap = self.create_group_swap(group)

    def __call__(self, sample: torch.tensor):
        self.swap(sample)

        return sample
    
  
    def swap_any(self, num_a, num_b, prob,):
        def swap(input: torch.tensor):
            ind_a = input == num_a
            ind_b = input == num_b

            count_a = ind_a.sum().item()
            count_b = ind_b.sum().item()

            ind_a[ind_a.clone()] = torch.multinomial(torch.tensor([1-prob, prob], device=input.device), count_a, replacement=True) == 1
            ind_b[ind_b.clone()] = torch.multinomial(torch.tensor([1-prob, prob], device=input.device), count_b, replacement=True) == 1

            input[ind_a] = num_b
            input[ind_b] = num_a
        
        return swap
    
    def create_group_swap(self, group:int):
        prob = 0.5
        match group:
            case 0:
                return self.swap_any(0,8, prob)
            case 1:
                return self.swap_any(1,7, prob)
            case 2:
                return self.swap_any(2,3, prob)
            case 3:
                return self.swap_any(4,5 ,prob)
            case 4:
                return self.swap_any(6,9, prob)