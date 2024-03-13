import gradio as gr


def inference(input_image, mask, model):
    from rembg import new_session, remove
    output = remove(
        input_image,
        session=new_session(model),
        only_mask=(True if mask == "Mask only" else False),
    )
    return output


title = "RemBG"
description = "Gradio demo for RemBG.https://github.com/danielgatis/rembg"


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            intput_image = gr.Image(type="pil", label="Input")
            mask = gr.Radio(
                ["Default", "Mask only"], type="value", value="Default", label="RemBG Choices"
            )
            model = gr.Dropdown(
                [
                    "u2net",
                    "u2netp",
                    "u2net_human_seg",
                    "u2net_cloth_seg",
                    "silueta",
                    "isnet-general-use",
                    "isnet-anime",
                    "sam",
                ],
                type="value",
                value="isnet-general-use",
                label="RemBG Models",
            )
        with gr.Column():
            output_image = gr.Image(type="pil", label="Output")

    btn = gr.Button(value="Run")
    btn.click(fn=inference, inputs=[intput_image, mask, model], outputs=output_image)

    gr.Examples(
         examples=[
            ["lion.png", "Default", "u2net"],
            ["girl.jpg", "Default", "u2net"],
            ["anime-girl.jpg", "Default", "isnet-anime"]
         ],
         inputs=[intput_image, mask, model]
    )


demo.launch(ssl_verify=False)
