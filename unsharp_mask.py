# Copyright (c) 2023 Jonathan S. Pollack (https://github.com/JPPhoto)


import numpy as np
from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    InputField,
    InvocationContext,
    WithMetadata,
    invocation,
)
from invokeai.app.invocations.primitives import ImageField, ImageOutput
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from PIL import Image, ImageFilter


@invocation("unsharp_mask", title="Unsharp Mask", tags=["unsharp_mask"], version="1.0.0")
class UnsharpMaskInvocation(BaseInvocation, WithMetadata):
    """Applies an unsharp mask filter to an image"""

    image: ImageField = InputField(description="The image to use")
    radius: float = InputField(gt=0, description="Unsharp mask radius", default=2)
    strength: float = InputField(ge=0, description="Unsharp mask strength", default=50)

    def pil_from_array(self, arr):
        return Image.fromarray((arr * 255).astype("uint8"))

    def array_from_pil(self, img):
        return np.array(img) / 255

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.services.images.get_pil_image(self.image.image_name)
        mode = image.mode

        alpha_channel = image.getchannel("A") if mode == "RGBA" else None
        image = image.convert("RGB")
        image_blurred = self.array_from_pil(image.filter(ImageFilter.GaussianBlur(radius=self.radius)))

        image = self.array_from_pil(image)
        image += (image - image_blurred) * (self.strength / 100.0)
        image = np.clip(image, 0, 1)
        image = self.pil_from_array(image)

        image = image.convert(mode)

        # Make the image RGBA if we had a source alpha channel
        if alpha_channel is not None:
            image.putalpha(alpha_channel)

        image_dto = context.services.images.create(
            image=image,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
            metadata=self.metadata,
            workflow=context.workflow,
        )

        return ImageOutput(
            image=ImageField(image_name=image_dto.image_name),
            width=image.width,
            height=image.height,
        )
