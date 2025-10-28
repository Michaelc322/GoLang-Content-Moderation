package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"time"

	pb "github.com/Michaelc322/GoLang-Content-Moderation/go-gateway/pb"
	"github.com/gin-gonic/gin"
	"google.golang.org/grpc"
)

func main() {
	// Connect to Python gRPC server
	conn, err := grpc.Dial("localhost:50051", grpc.WithInsecure(), grpc.WithBlock())
	if err != nil {
		log.Fatalf("Failed to connect to Python gRPC server: %v", err)
	}
	defer conn.Close()

	client := pb.NewModerationServiceClient(conn)
	router := gin.Default()

	router.GET("/healthz", func(c *gin.Context) {
		c.JSON(200, gin.H{"status": "ok"})
	})

	router.POST("/moderate/text", func(c *gin.Context) {
		var req struct {
			Text string `json:"text"`
		}
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		ctx, cancel := context.WithTimeout(context.Background(), time.Second*5)
		defer cancel()

		resp, err := client.PredictText(ctx, &pb.TextRequest{Text: req.Text})
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		c.JSON(http.StatusOK, gin.H{
			"toxicity_score": resp.ToxicityScore,
			"flagged":        resp.Flagged,
			"model_version":  resp.ModelVersion,
		})
	})

	fmt.Println("🚀 Go gateway running on port 8080")
	router.Run(":8080")
}
