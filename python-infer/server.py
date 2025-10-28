import grpc
from concurrent import futures
import moderation_pb2, moderation_pb2_grpc
from text_model import classify_text
from image_model import classify_image

class ModerationService(moderation_pb2_grpc.ModerationServiceServicer):
    def PredictText(self, request, context):
        score, flagged, version = classify_text(request.text)
        return moderation_pb2.TextResponse(
            toxicity_score=score,
            flagged=flagged,
            model_version=version
        )

    def PredictImage(self, request, context):
        score, flagged, version = classify_image(request.image_bytes)
        return moderation_pb2.ImageResponse(
            nsfw_score=score,
            flagged=flagged,
            model_version=version
        )

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    moderation_pb2_grpc.add_ModerationServiceServicer_to_server(ModerationService(), server)
    server.add_insecure_port("[::]:50051")
    print("🚀 gRPC Moderation server running on port 50051")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
