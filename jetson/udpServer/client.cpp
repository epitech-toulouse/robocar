#include <iostream>
#include <string>
#include <cstring>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <algorithm>

void print_usage(const char* prog_name) {
    std::cerr << "Usage: " << prog_name << " <server_ip> <port>" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        print_usage(argv[0]);
        return 1;
    }

    const char* server_ip = argv[1];
    int port = std::stoi(argv[2]);
    int sockfd;
    struct sockaddr_in server_addr;

    // Create socket
    if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        perror("socket creation failed");
        return 1;
    }

    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    
    if (inet_pton(AF_INET, server_ip, &server_addr.sin_addr) <= 0) {
        perror("Invalid address/ Address not supported");
        return 1;
    }

    // Send initial packet to register with server
    const char* hello = "Hello from client";
    sendto(sockfd, hello, strlen(hello), MSG_CONFIRM, (const struct sockaddr *)&server_addr, sizeof(server_addr));
    std::cout << "Connected to server at " << server_ip << ":" << port << std::endl;
    std::cout << "Waiting for messages..." << std::endl;

    char buffer[2048];

    while (true) {
        fd_set readfds;
        FD_ZERO(&readfds);
        FD_SET(STDIN_FILENO, &readfds);
        FD_SET(sockfd, &readfds);

        int max_fd = std::max(STDIN_FILENO, sockfd) + 1;

        int activity = select(max_fd, &readfds, NULL, NULL, NULL);

        if ((activity < 0) && (errno != EINTR)) {
            perror("select error");
            break;
        }

        // Handle socket input (Messages from server)
        if (FD_ISSET(sockfd, &readfds)) {
            struct sockaddr_in from_addr;
            socklen_t len = sizeof(from_addr);
            int n = recvfrom(sockfd, (char *)buffer, 2048, MSG_WAITALL, (struct sockaddr *)&from_addr, &len);
            
            if (n > 0) {
                std::cout.write(buffer, n);
                std::cout.flush();
            }
        }

        // Handle stdin input (Send to server)
        if (FD_ISSET(STDIN_FILENO, &readfds)) {
            std::string input;
            if (!std::getline(std::cin, input)) {
                break; // EOF
            }
            sendto(sockfd, input.c_str(), input.length(), MSG_CONFIRM, (const struct sockaddr *)&server_addr, sizeof(server_addr));
        }
    }

    close(sockfd);
    return 0;
}
