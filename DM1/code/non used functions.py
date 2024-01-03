##
#code for showing intenisty plotting maybe useful for later - various methods

plt.scatter(df.index, df.intensity)
plt.xlabel("index")
plt.ylabel("intensity")
plt.show()


sns.displot(x = "index",y="intensity", data= df_nostat,)


df_nostat.reset_index()[["intensity", "index"]].plot(kind = "scatter", y = "intensity" , x = "index")
# the reset index is for the problems with the new added index


df_nostat["intensity"].value_counts()

##
#code for plotting all the attributes
sns.pairplot(data=df_nostat, hue="sex")


