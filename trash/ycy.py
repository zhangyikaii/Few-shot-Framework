

        # def distribution_calibration_np(query, base_means, base_covs, k,alpha=0.21):
        #     base_means = base_means.cpu().detach().numpy()
        #     base_covs = base_covs.cpu().detach().numpy()
        #     dist = []
        #     for i in range(len(base_means)):
        #         dist.append(np.linalg.norm(query-base_means[i]))
        #     index = np.argpartition(dist, k)[:k]
        #     mean = np.concatenate([base_means[index], query[np.newaxis, :]])
        #     calibrated_mean = np.mean(mean, axis=0)
        #     calibrated_cov = np.mean(base_covs[index], axis=0)+alpha

        #     return calibrated_mean, calibrated_cov


                # support_data, query_data = s.cpu().detach().numpy(), q.cpu().detach().numpy()
                # support_label = support_label.cpu().detach().numpy()
                # query_label = query_label.cpu().detach().numpy()

                # sampled_data = []
                # sampled_label = []
                # num_sampled = int(750 / model.args.test_shot)

                # for i, v in enumerate(support_data):
                #     calibrated_mean, calibrated_cov = distribution_calibration_np(support_data[i], self.base_means, self.base_covs, k=2)
                #     sampled_data.append(np.random.multivariate_normal(mean=calibrated_mean, cov=calibrated_cov, size=num_sampled))
                #     sampled_label.extend([support_label[i]]*num_sampled)
                # sampled_data = np.concatenate([sampled_data[:]]).reshape(-1, 640)
                # print('sa', sampled_data.shape)
                # X_aug = np.concatenate([support_data, sampled_data])
                # Y_aug = np.concatenate([support_label, sampled_label])
                # classifier = LogisticRegression(max_iter=1000).fit(X=X_aug, y=Y_aug)

                # predicts = classifier.predict(query_data)
                # acc = np.mean(predicts == query_label)
                # print(acc)
                # return acc
