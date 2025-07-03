from model import resume_model

def evaluate(model_path: str, data_path: str, model_name: str = "Model", print_info: bool = False):
    model = resume_model(model_path=model_path)
    out_loss, der_loss, hes_loss, pde_loss = model.evaluate(data_path=data_path)
    if print_info:
        print(f"-------- {model_name} --------")
        print(f"OUT loss: {out_loss}")
        print(f"DER loss: {der_loss}")
        print(f"HES loss: {hes_loss}")
        print(f"PDE loss: {pde_loss}\n")
    return out_loss, der_loss, hes_loss, pde_loss