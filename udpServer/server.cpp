#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <set>
#include <algorithm>

// Comparator for sockaddr_in to be used in std::set
struct SockAddrComparator {
    bool operator()(const sockaddr_in& a, const sockaddr_in& b) const {
        if (a.sin_addr.s_addr != b.sin_addr.s_addr) {
            return a.sin_addr.s_addr < b.sin_addr.s_addr;
        }
        return a.sin_port < b.sin_port;
    }
};

void print_usage(const char* prog_name) {
    std::cerr << "Usage: " << prog_name << " <port>" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        print_usage(argv[0]);
        return 1;
    }

    int port = std::stoi(argv[1]);
    int sockfd;
    struct sockaddr_in server_addr;

    // Create socket
    if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        perror("socket creation failed");
        return 1;
    }

    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(port);

    // Bind
    if (bind(sockfd, (const struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        perror("bind failed");
        return 1;
    }

    std::cout << "UDP Server listening on port " << port << std::endl;
    std::cout << "Type messages and press Enter to broadcast to known clients." << std::endl;

    std::set<sockaddr_in, SockAddrComparator> clients;
    char buffer[1024];

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

        // Handle socket input (New clients / Data)
        if (FD_ISSET(sockfd, &readfds)) {
            struct sockaddr_in client_addr;
            socklen_t len = sizeof(client_addr);
            int n = recvfrom(sockfd, (char *)buffer, 1024, MSG_WAITALL, (struct sockaddr *)&client_addr, &len);
            
            if (n > 0) {
                buffer[n] = '\0';
                // Add to known clients
                if (clients.find(client_addr) == clients.end()) {
                    clients.insert(client_addr);
                    std::cout << "New client connected: " << inet_ntoa(client_addr.sin_addr) << ":" << ntohs(client_addr.sin_port) << std::endl;
                }
                
                // Print what was received
                std::cout << "Received from [" << inet_ntoa(client_addr.sin_addr) << ":" << ntohs(client_addr.sin_port) << "]: " << buffer << std::endl;
            }
        }

        // Handle stdin input (Broadcast)
        if (FD_ISSET(STDIN_FILENO, &readfds)) {
            std::string input;
            if (!std::getline(std::cin, input)) {
                break; // EOF
            }

            if (clients.empty()) {
                std::cout << "No clients connected. Message not sent." << std::endl;
            } else {
                for (const auto& client : clients) {
                    sendto(sockfd, input.c_str(), input.length(), MSG_CONFIRM, (const struct sockaddr *)&client, sizeof(client));
                }
                std::cout << "Broadcasted to " << clients.size() << " clients." << std::endl;
            }
        }
    }

    close(sockfd);
    return 0;
}
