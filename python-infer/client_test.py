import grpc
import moderation_pb2, moderation_pb2_grpc

channel = grpc.insecure_channel("localhost:50051")
stub = moderation_pb2_grpc.ModerationServiceStub(channel)

req = moderation_pb2.TextRequest(text="I love this project!")
resp = stub.PredictText(req)
print(resp)